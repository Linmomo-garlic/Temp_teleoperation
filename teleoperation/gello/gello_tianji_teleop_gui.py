#!/usr/bin/env python3
"""
Gello ↔ 天机 关节角遥操作 GUI
============================
两种互斥模式：
  - 主控从：本机读 COM9 Dynamixel，经 Thor 代理 set_joints 控制天机
  - 从控主：经 Thor get_state 读天机角，逆标定后写回 Gello 主臂位置

用法:
  conda activate new_gello
  python gello_tianji_teleop_gui.py
  # 或双击 run_teleop.bat
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gello_leader.agent_client import (  # noqa: E402
    AgentClient,
    ensure_agent_running,
)
from gello_leader.dynamixel.driver import (  # noqa: E402
    CURRENT_CONTROL_MODE,
    POSITION_CONTROL_MODE,
)
from gello_leader.dynamixel.multi_port_driver import (  # noqa: E402
    MultiPortDynamixelDriver,
)
from gello_leader.mapping import (  # noqa: E402
    JOINT_LIMITS_DEG,
    apply_calibration,
    clamp_tianji_deg,
    invert_calibration,
    rate_limit,
    to_tianji_deg,
    trajectory_smooth,
)

DEFAULT_CFG = os.path.join(ROOT, "configs", "teleop.yaml")
LOG_DIR = os.path.join(ROOT, "log")
LOG_FILE = os.path.join(LOG_DIR, "teleop_gui.log")

DIR_M2S = "master_to_slave"
DIR_S2M = "slave_to_master"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_offsets(path: str, offsets: List[float]) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("dynamixel", {})["joint_offsets_rad"] = [
        float(x) for x in offsets
    ]
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_yaml_patch(path: str, patch: Dict[str, Any]) -> None:
    """浅层合并写回 yaml（支持 dynamixel/tianji/control 一级段）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for section, values in patch.items():
        if isinstance(values, dict):
            data.setdefault(section, {}).update(values)
        else:
            data[section] = values
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


MODE_POSITION = "position"
MODE_IMPEDANCE = "impedance"
MODE_LABELS = {
    MODE_POSITION: "位置跟随",
    MODE_IMPEDANCE: "关节阻抗",
}

FORCE_SRC_MANUAL = "manual"
FORCE_SRC_SLAVE = "slave_ext"
FORCE_SRC_LABELS = {
    FORCE_SRC_MANUAL: "手动输入",
    FORCE_SRC_SLAVE: "从臂外力(joint_ext)",
}
# 推荐增益：天机外力常较大，映射到 Gello(≈1Nm) 宜偏小；近端略低、远端略高
DEFAULT_FORCE_GAINS = [0.10, 0.08, 0.10, 0.12, 0.15, 0.15, 0.18]


class TeleopApp:
    def __init__(self, root: tk.Tk, cfg_path: str):
        self.root = root
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)
        self._log_lock = threading.Lock()
        self._log_fp = self._open_log_file()

        self.driver: Optional[MultiPortDynamixelDriver] = None
        self.client: Optional[AgentClient] = None

        dx = self.cfg.get("dynamixel", {})
        self.offsets = [float(x) for x in dx.get("joint_offsets_rad", [0.0] * 7)]
        self.signs = [float(x) for x in dx.get("joint_signs", [1.0] * 7)]
        while len(self.offsets) < 7:
            self.offsets.append(0.0)
        while len(self.signs) < 7:
            self.signs.append(1.0)

        ctrl = self.cfg.get("control", {})
        self.freq_hz = float(ctrl.get("frequency_hz", 100))
        self.max_step = float(ctrl.get("max_step_deg", 8.0))
        self.smooth_enable = bool(ctrl.get("smooth_enable", True))
        self.max_vel_deg_s = float(ctrl.get("max_vel_deg_s", 120.0))
        self.max_acc_deg_s2 = float(ctrl.get("max_acc_deg_s2", 800.0))
        default_dir = str(ctrl.get("direction_default", DIR_M2S))
        if default_dir not in (DIR_M2S, DIR_S2M):
            default_dir = DIR_M2S

        tj = self.cfg.get("tianji", {})
        self.arm = str(tj.get("arm", "A")).upper()
        self.robot_ip = str(tj.get("robot_ip", "192.168.1.190"))
        self.vel_ratio = int(tj.get("vel_ratio", 20))
        self.acc_ratio = int(tj.get("acc_ratio", 20))
        mode = str(tj.get("control_mode", MODE_IMPEDANCE)).lower()
        if mode not in (MODE_POSITION, MODE_IMPEDANCE):
            mode = MODE_IMPEDANCE
        self.control_mode = mode
        self.joint_k = [float(x) for x in tj.get("joint_k", [5, 5, 5, 4, 3, 3, 2])]
        self.joint_d = [float(x) for x in tj.get("joint_d", [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])]
        while len(self.joint_k) < 7:
            self.joint_k.append(3.0)
        while len(self.joint_d) < 7:
            self.joint_d.append(0.2)

        thor = self.cfg.get("thor", {})
        self.thor_host = str(thor.get("host", "192.168.139.105"))
        self.agent_port = int(thor.get("agent_port", 15666))
        self.ssh_user = str(thor.get("ssh_user", "lambda2"))
        self.ssh_pass = str(thor.get("ssh_pass", "lambda"))
        self.auto_start = bool(thor.get("auto_start_agent", True))

        self.dry_run = tk.BooleanVar(value=bool(ctrl.get("dry_run_default", True)))
        self.direction = tk.StringVar(value=default_dir)
        self.freq_var = tk.StringVar(value=f"{self.freq_hz:.0f}")
        self.mode_var = tk.StringVar(value=MODE_LABELS.get(self.control_mode, "关节阻抗"))
        self.following = False
        self._stop_follow = threading.Event()
        self._follow_thread: Optional[threading.Thread] = None
        self._master_drive_enabled = False
        self._force_fb_enabled = False
        self._stop_force_fb = threading.Event()
        self._force_fb_thread: Optional[threading.Thread] = None
        self._last_cmd = [0.0] * 7
        self._last_vel = [0.0] * 7
        self._last_raw = np.zeros(7)
        self._lock = threading.Lock()
        self.max_force_nm = float(ctrl.get("max_force_nm", 0.8))
        self.force_scale = float(ctrl.get("force_scale", 1.0))
        self.force_duration_s = float(ctrl.get("force_duration_s", 3.0))
        src = str(ctrl.get("force_source", FORCE_SRC_MANUAL)).lower()
        if src not in (FORCE_SRC_MANUAL, FORCE_SRC_SLAVE):
            src = FORCE_SRC_MANUAL
        self.force_source = src
        self.force_gains = [float(x) for x in ctrl.get("force_gains", DEFAULT_FORCE_GAINS)]
        while len(self.force_gains) < 7:
            self.force_gains.append(DEFAULT_FORCE_GAINS[len(self.force_gains)])
        self.force_deadband_nm = float(ctrl.get("force_deadband_nm", 0.3))
        self.force_invert = bool(ctrl.get("force_invert", True))
        self._force_cmd_cache = [0.0] * 7
        self._force_cache_lock = threading.Lock()
        self._last_slave_ext = [0.0] * 7
        self.force_source_var = tk.StringVar(
            value=FORCE_SRC_LABELS.get(self.force_source, "手动输入")
        )
        self.force_invert_var = tk.BooleanVar(value=self.force_invert)
        self.force_deadband_var = tk.StringVar(value=f"{self.force_deadband_nm:.2f}")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._ui_tick)
        self._on_direction_changed(initial=True)
        self.log(f"日志文件 → {LOG_FILE}（每次启动覆盖）")

    # ---- UI ----

    def _build_ui(self) -> None:
        self.root.title("Gello ↔ 天机 关节角遥操作")
        self.root.geometry("860x700")
        self.root.minsize(720, 480)

        # 可滚动主区域（窗口矮时用滚轮/右侧条滑到下面）
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        page = ttk.Frame(canvas)
        page_id = canvas.create_window((0, 0), window=page, anchor="nw")

        def _on_page_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            canvas.itemconfigure(page_id, width=event.width)

        def _on_mousewheel(event) -> None:
            delta = int(-event.delta / 120) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        def _bind_wheel(_event=None) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_event=None) -> None:
            canvas.unbind_all("<MouseWheel>")

        page.bind("<Configure>", _on_page_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        page.bind("<Enter>", _bind_wheel)

        top = ttk.Frame(page, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="配置:").pack(side="left")
        self.cfg_var = tk.StringVar(value=self.cfg_path)
        ttk.Entry(top, textvariable=self.cfg_var, width=55).pack(side="left", padx=4)
        ttk.Button(top, text="重载", command=self.reload_cfg).pack(side="left")

        conn = ttk.LabelFrame(page, text="连接", padding=8)
        conn.pack(fill="x", padx=8, pady=4)

        row1 = ttk.Frame(conn)
        row1.pack(fill="x")
        ttk.Button(row1, text="连接主臂(COM)", command=self.connect_master).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="断开主臂", command=self.disconnect_master).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="启动/探测 Thor 代理", command=self.start_agent).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="连接代理", command=self.connect_agent).pack(
            side="left", padx=2
        )
        ttk.Button(row1, text="连接天机柜", command=self.connect_cabinet).pack(
            side="left", padx=2
        )

        row2 = ttk.Frame(conn)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="臂").pack(side="left")
        self.arm_var = tk.StringVar(value=self.arm)
        ttk.Combobox(
            row2, textvariable=self.arm_var, values=["A", "B"], width=4, state="readonly"
        ).pack(side="left", padx=4)
        ttk.Button(row2, text="使能", command=self.enable_arm).pack(side="left", padx=2)
        ttk.Button(row2, text="下使能", command=self.disable_arm).pack(
            side="left", padx=2
        )
        ttk.Button(row2, text="同步天机反馈→限速基准", command=self.sync_from_slave).pack(
            side="left", padx=2
        )
        self.dry_run_cb = ttk.Checkbutton(
            row2, text="干跑(不下发)", variable=self.dry_run
        )
        self.dry_run_cb.pack(side="left", padx=8)

        cal = ttk.LabelFrame(page, text="标定", padding=8)
        cal.pack(fill="x", padx=8, pady=4)
        ttk.Button(
            cal,
            text="当前姿设为零位(写 offsets)",
            command=self.capture_zero,
        ).pack(side="left", padx=2)
        ttk.Label(
            cal,
            text="先把主臂摆到与天机一致的 Home，再点此按钮；方向反了改 sign",
        ).pack(side="left", padx=8)

        ctrl = ttk.LabelFrame(page, text="遥操作", padding=8)
        ctrl.pack(fill="x", padx=8, pady=4)

        dir_row = ttk.Frame(ctrl)
        dir_row.pack(fill="x")
        ttk.Label(dir_row, text="控制方向:").pack(side="left")
        ttk.Radiobutton(
            dir_row,
            text="主控从 (Gello→天机)",
            value=DIR_M2S,
            variable=self.direction,
            command=self._on_direction_changed,
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            dir_row,
            text="从控主 (天机→Gello)",
            value=DIR_S2M,
            variable=self.direction,
            command=self._on_direction_changed,
        ).pack(side="left", padx=4)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="开始跟随", command=self.start_follow).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="停止跟随", command=self.stop_follow).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="急停", command=self.estop).pack(side="left", padx=2)
        self.status_var = tk.StringVar(value="状态: 未连接主臂")
        ttk.Label(btn_row, textvariable=self.status_var).pack(side="left", padx=12)

        param_row = ttk.Frame(ctrl)
        param_row.pack(fill="x", pady=(8, 0))
        ttk.Label(param_row, text="控制频率Hz").pack(side="left")
        ttk.Entry(param_row, textvariable=self.freq_var, width=6).pack(
            side="left", padx=4
        )
        ttk.Label(param_row, text="从臂模式").pack(side="left", padx=(10, 0))
        ttk.Combobox(
            param_row,
            textvariable=self.mode_var,
            values=[MODE_LABELS[MODE_POSITION], MODE_LABELS[MODE_IMPEDANCE]],
            width=10,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Button(
            param_row, text="应用并同步到Jetson", command=self.apply_control_params
        ).pack(side="left", padx=6)
        ttk.Label(
            param_row, text="频率同时调上位机跟随环与代理插值环；模式在下次使能生效"
        ).pack(side="left", padx=4)

        # ---- 定点关节角：经 Thor/Jetson 代理 set_joints 控天机 ----
        fixed = ttk.LabelFrame(
            page, text="定点关节角 → 天机 (经 Thor 代理)", padding=8
        )
        fixed.pack(fill="x", padx=8, pady=4)

        fj_row = ttk.Frame(fixed)
        fj_row.pack(fill="x")
        self.fixed_joint_vars: List[tk.StringVar] = []
        for i in range(7):
            ttk.Label(fj_row, text=f"J{i+1}").pack(side="left", padx=(0, 2))
            fv = tk.StringVar(value="0.0")
            self.fixed_joint_vars.append(fv)
            ttk.Entry(fj_row, textvariable=fv, width=7).pack(side="left", padx=(0, 6))

        fj_btn = ttk.Frame(fixed)
        fj_btn.pack(fill="x", pady=(6, 0))
        ttk.Label(fj_btn, text="插值(s)").pack(side="left")
        self.fixed_interp_var = tk.StringVar(value="2.0")
        ttk.Entry(fj_btn, textvariable=self.fixed_interp_var, width=6).pack(
            side="left", padx=4
        )
        ttk.Button(
            fj_btn, text="读天机反馈填入", command=self.fill_fixed_from_slave
        ).pack(side="left", padx=2)
        ttk.Button(fj_btn, text="填零位", command=self.fill_fixed_zeros).pack(
            side="left", padx=2
        )
        ttk.Button(
            fj_btn, text="读主臂标定目标填入", command=self.fill_fixed_from_master_cmd
        ).pack(side="left", padx=2)
        ttk.Button(
            fj_btn, text="下发到天机", command=self.send_fixed_joints
        ).pack(side="left", padx=8)
        ttk.Label(
            fj_btn, text="需先：连接代理→连接柜→使能；跟随中会先自动停止"
        ).pack(side="left", padx=4)

        # ---- 力反馈到主臂（电流力矩模式）----
        forcef = ttk.LabelFrame(
            page, text="力反馈 → 主臂 (手动Nm / 从臂joint_ext×增益)", padding=8
        )
        forcef.pack(fill="x", padx=8, pady=4)

        src_row = ttk.Frame(forcef)
        src_row.pack(fill="x")
        ttk.Label(src_row, text="力源").pack(side="left")
        ttk.Combobox(
            src_row,
            textvariable=self.force_source_var,
            values=[FORCE_SRC_LABELS[FORCE_SRC_MANUAL], FORCE_SRC_LABELS[FORCE_SRC_SLAVE]],
            width=18,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            src_row, text="取反(触感反射)", variable=self.force_invert_var
        ).pack(side="left", padx=6)
        ttk.Label(src_row, text="死区Nm").pack(side="left")
        ttk.Entry(src_row, textvariable=self.force_deadband_var, width=5).pack(
            side="left", padx=4
        )
        ttk.Button(
            src_row, text="读一次从臂外力→手动框", command=self.fill_force_from_slave_ext
        ).pack(side="left", padx=4)

        ff_row = ttk.Frame(forcef)
        ff_row.pack(fill="x", pady=(4, 0))
        ttk.Label(ff_row, text="力矩Nm").pack(side="left")
        self.force_vars: List[tk.StringVar] = []
        for i in range(7):
            ttk.Label(ff_row, text=f"J{i+1}").pack(side="left", padx=(4, 2))
            fv = tk.StringVar(value="0.0")
            self.force_vars.append(fv)
            ttk.Entry(ff_row, textvariable=fv, width=6).pack(side="left", padx=(0, 2))

        gain_row = ttk.Frame(forcef)
        gain_row.pack(fill="x", pady=(4, 0))
        ttk.Label(gain_row, text="增益").pack(side="left")
        self.force_gain_vars: List[tk.StringVar] = []
        for i in range(7):
            ttk.Label(gain_row, text=f"G{i+1}").pack(side="left", padx=(4, 2))
            gv = tk.StringVar(value=f"{self.force_gains[i]:.2f}")
            self.force_gain_vars.append(gv)
            ttk.Entry(gain_row, textvariable=gv, width=5).pack(side="left", padx=(0, 2))
        ttk.Button(
            gain_row, text="填推荐增益", command=self.fill_recommended_force_gains
        ).pack(side="left", padx=8)

        ext_row = ttk.Frame(forcef)
        ext_row.pack(fill="x", pady=(2, 0))
        ttk.Label(ext_row, text="从臂ext").pack(side="left")
        self.slave_ext_vars: List[tk.StringVar] = []
        for i in range(7):
            ev = tk.StringVar(value="--")
            self.slave_ext_vars.append(ev)
            ttk.Label(ext_row, textvariable=ev, width=6).pack(side="left", padx=2)

        ff_btn = ttk.Frame(forcef)
        ff_btn.pack(fill="x", pady=(6, 0))
        ttk.Label(ff_btn, text="缩放").pack(side="left")
        self.force_scale_var = tk.StringVar(value=f"{self.force_scale:.2f}")
        ttk.Entry(ff_btn, textvariable=self.force_scale_var, width=6).pack(
            side="left", padx=4
        )
        ttk.Label(ff_btn, text="限幅Nm").pack(side="left", padx=(8, 0))
        self.max_force_var = tk.StringVar(value=f"{self.max_force_nm:.2f}")
        ttk.Entry(ff_btn, textvariable=self.max_force_var, width=6).pack(
            side="left", padx=4
        )
        ttk.Label(ff_btn, text="反馈时间s").pack(side="left", padx=(8, 0))
        self.force_duration_var = tk.StringVar(value=f"{self.force_duration_s:.1f}")
        ttk.Entry(ff_btn, textvariable=self.force_duration_var, width=6).pack(
            side="left", padx=4
        )
        ttk.Button(ff_btn, text="清零", command=self.fill_force_zeros).pack(
            side="left", padx=2
        )
        ttk.Button(ff_btn, text="下发一次", command=self.apply_force_once).pack(
            side="left", padx=2
        )
        ttk.Button(
            ff_btn, text="开始力反馈", command=self.start_force_feedback
        ).pack(side="left", padx=2)
        ttk.Button(
            ff_btn, text="停止力反馈", command=self.stop_force_feedback
        ).pack(side="left", padx=2)
        self.force_status_var = tk.StringVar(value="力反馈: 关")
        ttk.Label(ff_btn, textvariable=self.force_status_var).pack(side="left", padx=8)
        ttk.Label(
            forcef,
            text="从臂外力模式：需连接柜并使能「关节阻抗」；tau=±gain×joint_ext→主臂电流模式；"
            "可与主控从跟随并行；时间0=手动停",
        ).pack(anchor="w", pady=(4, 0))

        table = ttk.LabelFrame(page, text="关节角 (°)", padding=8)
        table.pack(fill="x", padx=8, pady=4)

        hdr = ttk.Frame(table)
        hdr.pack(fill="x")
        self.col_raw_lbl = ttk.Label(hdr, text="主臂原始°", width=12)
        self.col_cmd_lbl = ttk.Label(hdr, text="标定目标°", width=12)
        for i, w in enumerate(
            [
                ttk.Label(hdr, text="关节", width=6),
                self.col_raw_lbl,
                self.col_cmd_lbl,
                ttk.Label(hdr, text="sign", width=12),
            ]
        ):
            w.grid(row=0, column=i, sticky="w")

        self.raw_vars = []
        self.cmd_vars = []
        self.sign_vars = []
        body = ttk.Frame(table)
        body.pack(fill="x")
        for i in range(7):
            ttk.Label(body, text=f"J{i+1}", width=6).grid(row=i, column=0, sticky="w")
            rv = tk.StringVar(value="--")
            cv = tk.StringVar(value="--")
            sv = tk.StringVar(value=str(self.signs[i]))
            self.raw_vars.append(rv)
            self.cmd_vars.append(cv)
            self.sign_vars.append(sv)
            ttk.Label(body, textvariable=rv, width=12).grid(row=i, column=1, sticky="w")
            ttk.Label(body, textvariable=cv, width=12).grid(row=i, column=2, sticky="w")
            e = ttk.Entry(body, textvariable=sv, width=8)
            e.grid(row=i, column=3, sticky="w")
            e.bind("<Return>", lambda _e, idx=i: self._apply_sign(idx))

        ttk.Button(body, text="应用全部 sign", command=self.apply_all_signs).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=6
        )

        logf = ttk.LabelFrame(page, text="日志", padding=4)
        logf.pack(fill="x", padx=8, pady=4)
        self.log_text = tk.Text(logf, height=6, wrap="word")
        self.log_text.pack(fill="x", expand=False)

    def _open_log_file(self):
        """每次启动覆盖写入 log/teleop_gui.log。"""
        os.makedirs(LOG_DIR, exist_ok=True)
        fp = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.write(f"=== Gello Tianji Teleop GUI 启动 {started} ===\n")
        fp.flush()
        return fp

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with self._log_lock:
            fp = self._log_fp
            if fp is not None and not fp.closed:
                try:
                    fp.write(line)
                    fp.flush()
                except Exception:
                    pass

        def _append():
            self.log_text.insert("end", line)
            self.log_text.see("end")

        self.root.after(0, _append)

    def _dir_label(self) -> str:
        return "主→从" if self.direction.get() == DIR_M2S else "从→主"

    def _on_direction_changed(self, initial: bool = False) -> None:
        was_following = self.following and not initial
        if was_following:
            self.stop_follow()

        if self.direction.get() == DIR_S2M:
            self.dry_run_cb.configure(text="干跑(不写主臂)")
            self.col_cmd_lbl.configure(text="天机反馈°")
            if self.driver is not None and not self._master_drive_enabled:
                # 从控主前保持松力矩，实控开始时再上力矩
                pass
        else:
            self.dry_run_cb.configure(text="干跑(不下发天机)")
            self.col_cmd_lbl.configure(text="标定目标°")
            self._disable_master_drive(release_only=True)

        if not initial:
            self.log(f"控制方向 → {self._dir_label()}")
            if was_following:
                self.log("切换方向已自动停止跟随")

    # ---- config / calib ----

    def reload_cfg(self) -> None:
        path = self.cfg_var.get().strip() or self.cfg_path
        self.cfg_path = path
        self.cfg = load_config(path)
        dx = self.cfg.get("dynamixel", {})
        self.offsets = [float(x) for x in dx.get("joint_offsets_rad", [0.0] * 7)]
        self.signs = [float(x) for x in dx.get("joint_signs", [1.0] * 7)]
        while len(self.offsets) < 7:
            self.offsets.append(0.0)
        while len(self.signs) < 7:
            self.signs.append(1.0)
        for i in range(7):
            self.sign_vars[i].set(str(self.signs[i]))
        ctrl = self.cfg.get("control", {})
        self.freq_hz = float(ctrl.get("frequency_hz", 100))
        self.max_step = float(ctrl.get("max_step_deg", 8.0))
        self.smooth_enable = bool(ctrl.get("smooth_enable", True))
        self.max_vel_deg_s = float(ctrl.get("max_vel_deg_s", 120.0))
        self.max_acc_deg_s2 = float(ctrl.get("max_acc_deg_s2", 800.0))
        self.freq_var.set(f"{self.freq_hz:.0f}")
        tj = self.cfg.get("tianji", {})
        mode = str(tj.get("control_mode", MODE_IMPEDANCE)).lower()
        if mode not in (MODE_POSITION, MODE_IMPEDANCE):
            mode = MODE_IMPEDANCE
        self.control_mode = mode
        self.mode_var.set(MODE_LABELS[self.control_mode])
        self.vel_ratio = int(tj.get("vel_ratio", self.vel_ratio))
        self.acc_ratio = int(tj.get("acc_ratio", self.acc_ratio))
        self.joint_k = [float(x) for x in tj.get("joint_k", self.joint_k)]
        self.joint_d = [float(x) for x in tj.get("joint_d", self.joint_d)]
        self.log(
            f"已重载配置 {path} | freq={self.freq_hz}Hz mode={self.control_mode} "
            f"smooth={self.smooth_enable} vmax={self.max_vel_deg_s} amax={self.max_acc_deg_s2}"
        )

    def _mode_key_from_label(self) -> str:
        label = self.mode_var.get().strip()
        for k, v in MODE_LABELS.items():
            if v == label:
                return k
        return MODE_IMPEDANCE

    def apply_control_params(self) -> None:
        """统一设置上位机频率，并同步到 Jetson 代理。"""
        try:
            freq = float(self.freq_var.get().strip())
        except ValueError:
            messagebox.showerror("控制参数", "频率非法")
            return
        freq = max(1.0, min(500.0, freq))
        mode = self._mode_key_from_label()
        self.freq_hz = freq
        self.control_mode = mode
        self.freq_var.set(f"{freq:.0f}")
        try:
            save_yaml_patch(
                self.cfg_path,
                {
                    "control": {
                        "frequency_hz": float(freq),
                        "max_step_deg": float(self.max_step),
                    },
                    "tianji": {
                        "control_mode": mode,
                        "joint_k": [float(x) for x in self.joint_k[:7]],
                        "joint_d": [float(x) for x in self.joint_d[:7]],
                    },
                },
            )
        except Exception as e:
            self.log(f"写配置失败: {e}")
        self.log(
            f"控制参数 → freq={freq:.0f}Hz mode={MODE_LABELS[mode]} "
            f"(上位机已更新；模式于下次使能生效)"
        )
        self._sync_params_to_agent()

    def _sync_params_to_agent(self) -> bool:
        if not self.client or not self.client.connected:
            self.log("代理未连接，频率/模式仅保存在本机，连上后再点「应用并同步」")
            return False
        try:
            r = self.client.call(
                "set_params",
                frequency_hz=float(self.freq_hz),
                control_mode=self.control_mode,
                max_step_deg=float(self.max_step),
                vel_ratio=int(self.vel_ratio),
                acc_ratio=int(self.acc_ratio),
                joint_k=[float(x) for x in self.joint_k[:7]],
                joint_d=[float(x) for x in self.joint_d[:7]],
                timeout=10.0,
            )
            self.log(f"已同步到 Jetson 代理: {r}")
            return bool(r.get("ok"))
        except Exception as e:
            self.log(f"同步代理参数失败: {e}")
            return False

    def _apply_sign(self, idx: int) -> None:
        try:
            self.signs[idx] = float(self.sign_vars[idx].get())
            self.log(f"J{idx+1} sign → {self.signs[idx]}")
        except ValueError:
            messagebox.showerror("错误", f"J{idx+1} sign 非法")

    def apply_all_signs(self) -> None:
        for i in range(7):
            self._apply_sign(i)
        try:
            save_yaml_patch(
                self.cfg_path,
                {"dynamixel": {"joint_signs": [float(x) for x in self.signs[:7]]}},
            )
            self.log(
                "joint_signs 已写入配置: "
                + ", ".join(str(float(x)) for x in self.signs[:7])
            )
        except Exception as e:
            self.log(f"写 joint_signs 失败: {e}")

    def capture_zero(self) -> None:
        if self.driver is None:
            messagebox.showwarning("提示", "请先连接主臂")
            return
        with self._lock:
            raw = self._last_raw.copy()
        self.offsets = [float(x) for x in raw.tolist()]
        try:
            save_offsets(self.cfg_path, self.offsets)
            self.log(
                "已将当前主臂姿设为零位，offsets 已写入 "
                + ", ".join(f"{x:.4f}" for x in self.offsets)
            )
        except Exception as e:
            self.log(f"写配置失败(内存已更新): {e}")

    # ---- master / agent ----

    def connect_master(self) -> None:
        if self.driver is not None:
            self.log("主臂已连接")
            return
        dx = self.cfg.get("dynamixel", {})
        ports = dx.get("ports", {})
        port_config = {str(k): list(v) for k, v in ports.items()}
        global_ids = list(dx.get("global_ids", [1, 2, 3, 4, 5, 6, 7]))
        servo_types = {int(k): str(v) for k, v in dx.get("servo_types", {}).items()}
        baud = int(dx.get("baudrate", 1000000))
        try:
            self.driver = MultiPortDynamixelDriver(
                port_config=port_config,
                global_ids=global_ids,
                servo_types_map=servo_types,
                baudrate=baud,
            )
            if self.cfg.get("control", {}).get("disable_master_torque", True):
                self.driver.set_torque_mode(False)
                self._master_drive_enabled = False
                self.log("主臂力矩已关闭")
            self.status_var.set("状态: 主臂已连接")
            self.log(f"主臂连接成功 ports={port_config} baud={baud}")
        except Exception as e:
            self.driver = None
            self.log(f"主臂连接失败: {e}")
            messagebox.showerror("主臂连接失败", str(e))

    def disconnect_master(self) -> None:
        self.stop_force_feedback()
        self.stop_follow()
        if self.driver is not None:
            try:
                self.driver.set_torque_mode(False)
            except Exception:
                pass
            try:
                self.driver.close()
            except Exception as e:
                self.log(f"关闭主臂异常: {e}")
            self.driver = None
            self._master_drive_enabled = False
            self.status_var.set("状态: 主臂已断开")
            self.log("主臂已断开")

    def start_agent(self) -> None:
        def _run():
            ok = ensure_agent_running(
                self.thor_host,
                self.agent_port,
                self.ssh_user,
                self.ssh_pass,
                self.log,
            )
            if not ok:
                self.log("请手动在 Thor 启动 thor_joint_agent.py")

        threading.Thread(target=_run, daemon=True).start()

    def connect_agent(self) -> None:
        try:
            if self.auto_start:
                ensure_agent_running(
                    self.thor_host,
                    self.agent_port,
                    self.ssh_user,
                    self.ssh_pass,
                    self.log,
                )
            c = AgentClient(self.thor_host, self.agent_port, timeout=5.0)
            c.connect()
            r = c.call("ping")
            if not r.get("ok"):
                raise RuntimeError(r.get("msg", "ping 失败"))
            if self.client:
                self.client.close()
            self.client = c
            self.log(f"已连接代理 {self.thor_host}:{self.agent_port}")
            self._sync_params_to_agent()
        except Exception as e:
            self.log(f"连接代理失败: {e}")
            messagebox.showerror("代理", str(e))

    def connect_cabinet(self) -> None:
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理")
            return
        try:
            # 经 Thor 代理跳连天机柜( eth1 → robot_ip )，SDK 建链可能较慢
            self.log(f"经 Thor 连接天机柜 {self.robot_ip} ...")
            r = self.client.call("connect", ip=self.robot_ip, timeout=30.0)
            self.log(f"连接柜: {r}")
            if not r.get("ok"):
                messagebox.showerror("天机柜", r.get("msg", "失败"))
        except Exception as e:
            self.log(
                f"连接柜失败: {e}（本机不直连天机；请在 Thor 上 ping {self.robot_ip}）"
            )
            messagebox.showerror("天机柜", str(e))

    def enable_arm(self) -> None:
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理")
            return
        arm = self.arm_var.get().upper()
        mode = self._mode_key_from_label()
        self.control_mode = mode
        try:
            self._sync_params_to_agent()
            label = MODE_LABELS.get(mode, mode)
            self.log(f"使能 {arm} ({label}) 中...")
            r = self.client.call(
                "enable",
                arm=arm,
                vel=self.vel_ratio,
                acc=self.acc_ratio,
                mode=mode,
                K=[float(x) for x in self.joint_k[:7]],
                D=[float(x) for x in self.joint_d[:7]],
                timeout=30.0,
            )
            self.log(f"使能 {arm}: {r}")
            if not r.get("ok"):
                messagebox.showerror("使能", r.get("msg", "失败"))
            else:
                self.status_var.set(f"状态: {arm} 已使能/{label}")
        except Exception as e:
            self.log(f"使能失败: {e}")
            messagebox.showerror("使能", str(e))

    def disable_arm(self) -> None:
        if not self.client or not self.client.connected:
            return
        arm = self.arm_var.get().upper()
        try:
            r = self.client.call("disable", arm=arm, timeout=15.0)
            self.log(f"下使能 {arm}: {r}")
        except Exception as e:
            self.log(f"下使能失败: {e}")

    def _parse_slave_joints_deg(self, r: Dict[str, Any]) -> Optional[List[float]]:
        snap = r.get("state") if isinstance(r.get("state"), dict) else r
        arms = (snap or {}).get("arms") or {}
        arm = self.arm_var.get().upper()
        info = arms.get(arm) or {}
        fb = info.get("joint_pos_e") or info.get("joint_pos") or info.get("target")
        if not fb or len(fb) < 7:
            return None
        return [float(x) for x in fb[:7]]

    def sync_from_slave(self) -> None:
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理")
            return
        arm = self.arm_var.get().upper()
        try:
            self.client.call("sync_target", arm=arm, timeout=15.0)
            r = self.client.call("get_state", timeout=15.0)
            fb = self._parse_slave_joints_deg(r)
            if fb is None:
                self.log(f"无法读取反馈: {r}")
                return
            self._last_cmd = list(fb)
            self._last_vel = [0.0] * 7
            self.log(
                "限速/平滑基准已同步为天机当前角: "
                + ", ".join(f"{x:.1f}" for x in self._last_cmd)
            )
        except Exception as e:
            self.log(f"同步失败: {e}")

    # ---- 定点关节角 (经 Thor/Jetson thor_joint_agent.set_joints) ----

    def _read_fixed_joints_deg(self) -> Optional[List[float]]:
        vals: List[float] = []
        for i, var in enumerate(self.fixed_joint_vars):
            try:
                vals.append(float(var.get().strip()))
            except ValueError:
                messagebox.showerror("定点关节", f"J{i+1} 不是合法数字")
                return None
        return clamp_tianji_deg(vals, JOINT_LIMITS_DEG)

    def fill_fixed_zeros(self) -> None:
        for var in self.fixed_joint_vars:
            var.set("0.0")
        self.log("定点关节角已填零位")

    def fill_fixed_from_slave(self) -> None:
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理")
            return
        try:
            r = self.client.call("get_state", timeout=15.0)
            fb = self._parse_slave_joints_deg(r)
            if fb is None:
                self.log(f"读天机反馈失败: {r}")
                messagebox.showerror("定点关节", "无法解析天机反馈")
                return
            for i, v in enumerate(fb):
                self.fixed_joint_vars[i].set(f"{v:.3f}")
            self.log(
                "定点关节角 ← 天机反馈: " + ", ".join(f"{x:.2f}" for x in fb)
            )
        except Exception as e:
            self.log(f"读天机反馈失败: {e}")
            messagebox.showerror("定点关节", str(e))

    def fill_fixed_from_master_cmd(self) -> None:
        if self.driver is None:
            messagebox.showwarning("提示", "请先连接主臂")
            return
        try:
            cmd = self._read_cmd_deg()
            for i, v in enumerate(cmd):
                self.fixed_joint_vars[i].set(f"{v:.3f}")
            self.log(
                "定点关节角 ← 主臂标定目标: " + ", ".join(f"{x:.2f}" for x in cmd)
            )
        except Exception as e:
            self.log(f"读主臂标定目标失败: {e}")
            messagebox.showerror("定点关节", str(e))

    def send_fixed_joints(self) -> None:
        """经 Thor 代理调用 Jetson 侧 set_joints，下发固定关节角到天机。"""
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理与天机柜")
            return
        joints = self._read_fixed_joints_deg()
        if joints is None:
            return
        try:
            interp_s = float(self.fixed_interp_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("定点关节", "插值时间非法")
            return
        interp_s = max(0.0, min(interp_s, 30.0))
        arm = self.arm_var.get().upper()

        if self.following:
            self.stop_follow()
            self.log("定点下发前已停止跟随")

        if not messagebox.askyesno(
            "确认 · 定点下发",
            f"经 Thor 控制天机臂 {arm}\n"
            f"目标(°): {', '.join(f'{x:.2f}' for x in joints)}\n"
            f"插值: {interp_s:.2f}s\n"
            "请确认周围安全。继续？",
        ):
            return

        def _run() -> None:
            try:
                self.log(
                    f"定点下发 {arm} interp={interp_s:.2f}s → "
                    + ", ".join(f"{x:.2f}" for x in joints)
                )
                # 插值期间代理占锁，超时需覆盖整段运动
                timeout = max(15.0, interp_s + 10.0)
                r = self.client.call(
                    "set_joints",
                    arm=arm,
                    joints=joints,
                    interp_s=interp_s,
                    timeout=timeout,
                )
                self.log(f"定点下发结果: {r}")
                if r.get("ok"):
                    self._last_cmd = list(joints)
                    self._last_vel = [0.0] * 7

                    def _ok():
                        self.status_var.set(f"状态: 定点到位 {arm}")

                    self.root.after(0, _ok)
                else:
                    msg = r.get("msg", "下发失败")
                    self.root.after(
                        0, lambda m=msg: messagebox.showerror("定点关节", m)
                    )
            except Exception as e:
                self.log(f"定点下发失败: {e}")
                self.root.after(
                    0, lambda err=str(e): messagebox.showerror("定点关节", err)
                )

        threading.Thread(target=_run, daemon=True).start()

    # ---- 力反馈 → 主臂 ----

    def _force_source_key(self) -> str:
        label = self.force_source_var.get().strip()
        for k, v in FORCE_SRC_LABELS.items():
            if v == label:
                return k
        return FORCE_SRC_MANUAL

    def fill_force_zeros(self) -> None:
        for var in self.force_vars:
            var.set("0.0")
        self.log("力反馈输入已清零")

    def fill_recommended_force_gains(self) -> None:
        for i, g in enumerate(DEFAULT_FORCE_GAINS):
            self.force_gain_vars[i].set(f"{g:.2f}")
            self.force_gains[i] = float(g)
        self.log(
            "已填推荐增益: " + ", ".join(f"{g:.2f}" for g in DEFAULT_FORCE_GAINS)
            + " (近端偏小/远端略大；再按手感微调)"
        )

    def _read_force_gains(self, *, silent: bool = False) -> Optional[List[float]]:
        gains: List[float] = []
        for i, var in enumerate(self.force_gain_vars):
            try:
                g = float(var.get().strip())
            except ValueError:
                if not silent:
                    messagebox.showerror("力反馈", f"G{i+1} 增益非法")
                return None
            g = max(0.0, min(5.0, g))
            gains.append(g)
        self.force_gains = gains
        return gains

    def _parse_slave_joint_ext(self, r: Dict[str, Any]) -> Optional[List[float]]:
        snap = r.get("state") if isinstance(r.get("state"), dict) else r
        arms = (snap or {}).get("arms") or {}
        arm = self.arm_var.get().upper()
        info = arms.get(arm) or {}
        ext = info.get("joint_ext")
        if not ext or len(ext) < 7:
            return None
        return [float(x) for x in ext[:7]]

    def _fetch_slave_joint_ext(self, *, timeout: float = 5.0) -> Optional[List[float]]:
        if not self.client or not self.client.connected:
            return None
        r = self.client.call("get_state", timeout=timeout)
        return self._parse_slave_joint_ext(r)

    def fill_force_from_slave_ext(self) -> None:
        if not self.client or not self.client.connected:
            messagebox.showwarning("提示", "请先连接代理与天机柜")
            return
        try:
            ext = self._fetch_slave_joint_ext(timeout=10.0)
            if ext is None:
                messagebox.showerror("力反馈", "无法读取从臂 joint_ext")
                return
            self._last_slave_ext = list(ext)
            for i, v in enumerate(ext):
                self.force_vars[i].set(f"{v:.3f}")
                self.slave_ext_vars[i].set(f"{v:.2f}")
            self.log(
                "从臂 joint_ext → 手动框: " + ", ".join(f"{x:.3f}" for x in ext)
            )
        except Exception as e:
            self.log(f"读从臂外力失败: {e}")
            messagebox.showerror("力反馈", str(e))

    def _slave_ext_to_cmd(
        self, ext: Sequence[float], gains: Sequence[float], scale: float, deadband: float
    ) -> List[float]:
        sign = -1.0 if self.force_invert_var.get() else 1.0
        out: List[float] = []
        for i in range(7):
            e = float(ext[i])
            if abs(e) < deadband:
                e = 0.0
            v = sign * float(gains[i]) * e * scale
            v = max(-self.max_force_nm, min(self.max_force_nm, v))
            out.append(v)
        return out

    def _compute_force_cmd_nm(self, *, silent: bool = False) -> Optional[List[float]]:
        """按当前力源计算下发到主臂前的天机坐标系力矩命令。"""
        try:
            scale = float(self.force_scale_var.get().strip() or "1")
            lim = float(self.max_force_var.get().strip() or "0.8")
            deadband = float(self.force_deadband_var.get().strip() or "0")
        except ValueError:
            if not silent:
                messagebox.showerror("力反馈", "缩放/限幅/死区非法")
            return None
        self.force_scale = scale
        self.max_force_nm = max(0.01, min(abs(lim), 3.0))
        self.force_deadband_nm = max(0.0, min(abs(deadband), 20.0))
        self.force_invert = bool(self.force_invert_var.get())
        gains = self._read_force_gains(silent=silent)
        if gains is None:
            return None

        src = self._force_source_key()
        self.force_source = src
        if src == FORCE_SRC_SLAVE:
            try:
                ext = self._fetch_slave_joint_ext(timeout=2.0)
            except Exception as e:
                if not silent:
                    self.log(f"读从臂外力失败: {e}")
                return None
            if ext is None:
                if not silent:
                    messagebox.showerror("力反馈", "无法读取从臂 joint_ext")
                return None
            self._last_slave_ext = list(ext)
            vals = self._slave_ext_to_cmd(
                ext, gains, scale, self.force_deadband_nm
            )
        else:
            vals = []
            for i, var in enumerate(self.force_vars):
                try:
                    v = float(var.get().strip()) * scale
                except ValueError:
                    if not silent:
                        messagebox.showerror("力反馈", f"J{i+1} 力矩不是合法数字")
                    return None
                if self.force_invert_var.get():
                    v = -v
                v = max(-self.max_force_nm, min(self.max_force_nm, v))
                vals.append(v)

        with self._force_cache_lock:
            self._force_cmd_cache = list(vals)
        return vals

    def _read_force_cmd_nm(self, *, silent: bool = False) -> Optional[List[float]]:
        return self._compute_force_cmd_nm(silent=silent)

    def _forces_to_master_nm(self, forces_cmd: Sequence[float]) -> List[float]:
        """天机侧力矩 → 主臂舵机力矩：tau_raw = tau_cmd * sign。"""
        out: List[float] = []
        for i in range(7):
            s = float(self.signs[i]) if abs(float(self.signs[i])) > 1e-9 else 1.0
            out.append(float(forces_cmd[i]) * s)
        return out

    def _enter_master_current_mode(self) -> None:
        assert self.driver is not None
        self.driver.set_torque_mode(False)
        self.driver.set_operating_mode(CURRENT_CONTROL_MODE)
        self.driver.set_torque_mode(True)
        self._master_drive_enabled = False  # 非位置驱动
        self.log("主臂已进入电流力矩模式并开启力矩")

    def _apply_master_torques(self, forces_cmd: Sequence[float]) -> None:
        assert self.driver is not None
        tau = self._forces_to_master_nm(forces_cmd)
        self.driver.set_torque(tau)

    def apply_force_once(self) -> None:
        if self.driver is None:
            messagebox.showwarning("提示", "请先连接主臂")
            return
        if self._force_source_key() == FORCE_SRC_SLAVE:
            if not self.client or not self.client.connected:
                messagebox.showwarning("提示", "从臂外力模式请先连接代理")
                return
        forces = self._compute_force_cmd_nm()
        if forces is None:
            return
        try:
            if self.following and self.direction.get() == DIR_S2M:
                self.stop_follow()
            if not self._force_fb_enabled:
                self._enter_master_current_mode()
            self._apply_master_torques(forces)
            self.log(
                f"力反馈单次({FORCE_SRC_LABELS[self.force_source]}) Nm: "
                + ", ".join(f"{x:.3f}" for x in forces)
            )
            self.force_status_var.set("力反馈: 已单次下发")
        except Exception as e:
            self.log(f"力反馈下发失败: {e}")
            messagebox.showerror("力反馈", str(e))

    def start_force_feedback(self) -> None:
        if self.driver is None:
            messagebox.showwarning("提示", "请先连接主臂")
            return
        if self._force_fb_enabled:
            return
        src = self._force_source_key()
        if src == FORCE_SRC_SLAVE:
            if not self.client or not self.client.connected:
                messagebox.showwarning("提示", "从臂外力模式请先连接代理与天机柜")
                return
            if self.control_mode != MODE_IMPEDANCE and self._mode_key_from_label() != MODE_IMPEDANCE:
                if not messagebox.askyesno(
                    "模式提示",
                    "推荐从臂使用「关节阻抗」后再开外力反馈。\n"
                    "当前配置不是关节阻抗，仍继续？",
                ):
                    return
        forces = self._compute_force_cmd_nm()
        if forces is None:
            return
        try:
            duration_s = float(self.force_duration_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror("力反馈", "反馈时间非法")
            return
        if duration_s < 0:
            messagebox.showerror("力反馈", "反馈时间不能为负")
            return
        duration_s = min(duration_s, 600.0)
        self.force_duration_s = duration_s
        # 从控主占位置模式，必须停；主控从可与从臂外力反馈并行
        if self.following and self.direction.get() == DIR_S2M:
            self.stop_follow()
            self.log("力反馈开始前已停止从控主跟随")
        elif self.following and src == FORCE_SRC_MANUAL:
            self.stop_follow()
            self.log("手动力反馈开始前已停止跟随")
        dur_txt = "手动停止前一直输出" if duration_s <= 0 else f"{duration_s:.1f} s 后自动停止"
        if not messagebox.askyesno(
            "确认 · 力反馈",
            f"力源: {FORCE_SRC_LABELS[src]}\n"
            "主臂将进入电流模式并输出力矩。\n"
            f"限幅 ±{self.max_force_nm:.2f} Nm，缩放 {self.force_scale:.2f}\n"
            f"增益: {', '.join(f'{g:.2f}' for g in self.force_gains)}\n"
            f"取反={self.force_invert_var.get()} 死区={self.force_deadband_nm:.2f}Nm\n"
            f"反馈时间: {dur_txt}\n"
            "请握稳主臂、周围无干涉。继续？",
        ):
            return
        try:
            self._enter_master_current_mode()
            self._apply_master_torques(forces)
        except Exception as e:
            self.log(f"力反馈启动失败: {e}")
            messagebox.showerror("力反馈", str(e))
            return

        self._stop_force_fb.clear()
        self._force_fb_enabled = True
        if duration_s > 0:
            self.force_status_var.set(f"力反馈: 运行中 ({duration_s:.1f}s)")
        else:
            self.force_status_var.set("力反馈: 运行中 (无时限)")
        self.status_var.set("状态: 力反馈运行中")
        self.log(
            f"开始力反馈({FORCE_SRC_LABELS[src]}) → 主臂 | "
            + ", ".join(f"{x:.3f}" for x in forces)
            + f" | {self.freq_hz:.0f}Hz | duration={duration_s:.1f}s"
        )

        def _loop() -> None:
            period = 1.0 / max(1.0, self.freq_hz)
            t_end = (
                time.perf_counter() + duration_s if duration_s > 0 else None
            )
            last_status = 0.0
            while not self._stop_force_fb.is_set():
                if t_end is not None and time.perf_counter() >= t_end:
                    self.log(f"力反馈时间到 ({duration_s:.1f}s)，自动停止")
                    self.root.after(0, self.stop_force_feedback)
                    break
                t0 = time.perf_counter()
                try:
                    if src == FORCE_SRC_SLAVE:
                        cmd = self._compute_force_cmd_nm(silent=True)
                        if cmd is None:
                            with self._force_cache_lock:
                                cmd = list(self._force_cmd_cache)
                    else:
                        with self._force_cache_lock:
                            cmd = list(self._force_cmd_cache)
                    if self.driver is not None:
                        self._apply_master_torques(cmd)
                    now = time.perf_counter()
                    if now - last_status >= 0.25:
                        last_status = now
                        ext = list(self._last_slave_ext)
                        remain = (
                            max(0.0, t_end - now) if t_end is not None else None
                        )

                        def _upd(e=ext, r=remain, s=src):
                            for i in range(7):
                                self.slave_ext_vars[i].set(f"{e[i]:.2f}")
                            if r is not None:
                                self.force_status_var.set(
                                    f"力反馈: 运行中 (剩余{r:.1f}s)"
                                )
                            elif s == FORCE_SRC_SLAVE:
                                self.force_status_var.set(
                                    "力反馈: 从臂外力运行中"
                                )

                        self.root.after(0, _upd)
                except Exception as e:
                    self.log(f"力反馈环错误: {e}")
                    time.sleep(0.2)
                dt = time.perf_counter() - t0
                time.sleep(max(0.0, period - dt))

        self._force_fb_thread = threading.Thread(target=_loop, daemon=True)
        self._force_fb_thread.start()

    def stop_force_feedback(self) -> None:
        if not self._force_fb_enabled and (
            self._force_fb_thread is None or not self._force_fb_thread.is_alive()
        ):
            # 单次下发后也可松力矩
            if self.driver is not None:
                try:
                    self.driver.set_torque_mode(False)
                except Exception:
                    pass
            self.force_status_var.set("力反馈: 关")
            return
        self._stop_force_fb.set()
        self._force_fb_enabled = False
        if self._force_fb_thread and self._force_fb_thread.is_alive():
            self._force_fb_thread.join(timeout=1.5)
        self._force_fb_thread = None
        if self.driver is not None:
            try:
                # 清零电流再松力矩
                try:
                    self.driver.set_torque([0.0] * 7)
                except Exception:
                    pass
                self.driver.set_torque_mode(False)
            except Exception as e:
                self.log(f"停止力反馈关力矩异常: {e}")
        self.force_status_var.set("力反馈: 关")
        self.log("已停止力反馈，主臂力矩已关")

    def _enable_master_drive(self) -> None:
        assert self.driver is not None
        if self._force_fb_enabled:
            self.stop_force_feedback()
        cur = np.asarray(self.driver.get_joints(), dtype=float).reshape(-1)
        self.driver.set_torque_mode(False)
        self.driver.set_operating_mode(POSITION_CONTROL_MODE)
        self.driver.set_torque_mode(True)
        self.driver.set_joints(cur.tolist())
        with self._lock:
            self._last_raw = cur.copy()
        self._master_drive_enabled = True
        self.log("主臂已进入位置模式并开启力矩")

    def _disable_master_drive(self, release_only: bool = False) -> None:
        if self.driver is None:
            self._master_drive_enabled = False
            return
        if not self._master_drive_enabled and release_only:
            return
        try:
            self.driver.set_torque_mode(False)
            if self._master_drive_enabled:
                self.log("主臂力矩已关闭")
        except Exception as e:
            self.log(f"关闭主臂力矩异常: {e}")
        self._master_drive_enabled = False

    # ---- follow loop ----

    def start_follow(self) -> None:
        if self.driver is None:
            messagebox.showwarning("提示", "请先连接主臂")
            return
        if self.following:
            return
        if self._force_fb_enabled:
            # 从臂外力反馈可与主控从并行；手动力反馈/从控主需停力反馈
            if (
                self._force_source_key() == FORCE_SRC_MANUAL
                or self.direction.get() == DIR_S2M
            ):
                self.stop_force_feedback()
                self.log("开始跟随前已停止力反馈")

        direction = self.direction.get()
        dry = self.dry_run.get()

        if direction == DIR_S2M:
            if not self.client or not self.client.connected:
                messagebox.showwarning("提示", "从控主需要先连接代理与天机柜")
                return
            if not dry:
                if not messagebox.askyesno(
                    "确认 · 从控主",
                    f"将以约 {self.freq_hz} Hz 用天机臂 {self.arm_var.get()} "
                    "关节角驱动 Gello 主臂。\n"
                    "请确认主从已对齐、主臂周围无干涉，急停可用。\n继续？",
                ):
                    return
                try:
                    self._enable_master_drive()
                except Exception as e:
                    self.log(f"主臂进入位置模式失败: {e}")
                    messagebox.showerror("主臂", str(e))
                    return
            else:
                self._disable_master_drive(release_only=True)
        else:
            # 主控从：确保主臂松力矩便于手掰
            self._disable_master_drive(release_only=False)
            if self.driver is not None:
                try:
                    self.driver.set_torque_mode(False)
                except Exception:
                    pass
            if not dry:
                if not self.client or not self.client.connected:
                    messagebox.showwarning("提示", "非干跑需要先连接代理并使能")
                    return
                if not messagebox.askyesno(
                    "确认 · 主控从",
                    f"将以约 {self.freq_hz} Hz 向臂 {self.arm_var.get()} 下发关节角。\n"
                    "请确认主从已对齐 Home，周围安全。\n继续？",
                ):
                    return

        self._stop_follow.clear()
        self.following = True
        self._last_vel = [0.0] * 7
        target = (
            self._follow_loop_slave_to_master
            if direction == DIR_S2M
            else self._follow_loop_master_to_slave
        )
        self._follow_thread = threading.Thread(target=target, daemon=True)
        self._follow_thread.start()
        mode = "干跑" if dry else "实控"
        smooth = (
            f" | 发送端平滑 vmax={self.max_vel_deg_s} amax={self.max_acc_deg_s2}"
            if direction == DIR_M2S and self.smooth_enable
            else ""
        )
        self.status_var.set(f"状态: 跟随中 ({self._dir_label()} / {mode})")
        self.log(f"开始跟随 ({self._dir_label()} / {mode}){smooth}")

    def stop_follow(self) -> None:
        if not self.following:
            # 即使未跟随，从控主实控残留力矩也要松
            if self._master_drive_enabled:
                self._disable_master_drive()
            return
        self._stop_follow.set()
        self.following = False
        if self._follow_thread and self._follow_thread.is_alive():
            self._follow_thread.join(timeout=1.5)
        self._follow_thread = None
        if self.direction.get() == DIR_S2M or self._master_drive_enabled:
            self._disable_master_drive()
        self.status_var.set("状态: 已停止跟随")
        self.log("已停止跟随")

    def estop(self) -> None:
        self.stop_force_feedback()
        self.stop_follow()
        if self.driver is not None:
            try:
                self.driver.set_torque_mode(False)
                self._master_drive_enabled = False
                self.log("急停: 主臂力矩已关")
            except Exception as e:
                self.log(f"急停关主臂力矩失败: {e}")
        if self.client and self.client.connected:
            try:
                r = self.client.call("soft_stop")
                self.log(f"急停(soft_stop): {r}")
            except Exception as e:
                self.log(f"急停调用失败: {e}")
        self.status_var.set("状态: 急停")
        self.force_status_var.set("力反馈: 关")

    def _read_cmd_deg(self) -> List[float]:
        assert self.driver is not None
        raw = self.driver.get_joints()
        with self._lock:
            self._last_raw = np.asarray(raw, dtype=float).copy()
        q = apply_calibration(raw, self.offsets, self.signs)
        return to_tianji_deg(q, JOINT_LIMITS_DEG)

    def _follow_loop_master_to_slave(self) -> None:
        period = 1.0 / max(1.0, self.freq_hz)
        while not self._stop_follow.is_set():
            t0 = time.perf_counter()
            try:
                target = self._read_cmd_deg()
                if self.smooth_enable:
                    cmd, vel = trajectory_smooth(
                        target,
                        self._last_cmd,
                        self._last_vel,
                        period,
                        self.max_vel_deg_s,
                        self.max_acc_deg_s2,
                    )
                    # 硬限幅兜底；若被截断则按实际步长回写速度状态
                    cmd = rate_limit(cmd, self._last_cmd, self.max_step)
                    for i in range(7):
                        actual = cmd[i] - self._last_cmd[i]
                        if abs(actual) < 1e-12:
                            vel[i] = 0.0
                        elif abs(vel[i] * period) > abs(actual) + 1e-9:
                            vel[i] = actual / period
                    self._last_vel = vel
                else:
                    cmd = rate_limit(target, self._last_cmd, self.max_step)
                    self._last_vel = [0.0] * 7
                self._last_cmd = list(cmd)
                if not self.dry_run.get() and self.client and self.client.connected:
                    arm = self.arm_var.get().upper()
                    self.client.call(
                        "set_joints", arm=arm, joints=cmd, interp_s=0.0
                    )
            except Exception as e:
                self.log(f"跟随环错误(主→从): {e}")
                time.sleep(0.2)
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    def _follow_loop_slave_to_master(self) -> None:
        period = 1.0 / max(1.0, self.freq_hz)
        while not self._stop_follow.is_set():
            t0 = time.perf_counter()
            try:
                if not self.client or not self.client.connected:
                    raise RuntimeError("代理未连接")
                r = self.client.call("get_state")
                fb = self._parse_slave_joints_deg(r)
                if fb is None:
                    raise RuntimeError(f"无法解析天机反馈: {r}")
                cmd = clamp_tianji_deg(fb, JOINT_LIMITS_DEG)
                cmd = rate_limit(cmd, self._last_cmd, self.max_step)
                self._last_cmd = list(cmd)

                q_cmd_rad = np.deg2rad(np.asarray(cmd, dtype=float))
                raw = invert_calibration(q_cmd_rad, self.offsets, self.signs)

                if self.driver is not None:
                    # 干跑也刷新实测；实控才写目标
                    measured = self.driver.get_joints()
                    with self._lock:
                        self._last_raw = np.asarray(measured, dtype=float).copy()
                    if not self.dry_run.get() and self._master_drive_enabled:
                        self.driver.set_joints(raw.tolist())
            except Exception as e:
                self.log(f"跟随环错误(从→主): {e}")
                time.sleep(0.2)
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    def _ui_tick(self) -> None:
        try:
            if self.driver is not None:
                if not self.following:
                    if self.direction.get() == DIR_M2S:
                        cmd = self._read_cmd_deg()
                    else:
                        # 从控主空闲：读主臂，并尽量刷天机反馈到目标列
                        raw = self.driver.get_joints()
                        with self._lock:
                            self._last_raw = np.asarray(raw, dtype=float).copy()
                        cmd = list(self._last_cmd)
                        if self.client and self.client.connected:
                            try:
                                fb = self._parse_slave_joints_deg(
                                    self.client.call("get_state")
                                )
                                if fb is not None:
                                    cmd = clamp_tianji_deg(fb, JOINT_LIMITS_DEG)
                                    self._last_cmd = list(cmd)
                            except Exception:
                                pass
                else:
                    cmd = list(self._last_cmd)
                with self._lock:
                    raw = self._last_raw.copy()
                raw_deg = np.rad2deg(raw)
                for i in range(7):
                    self.raw_vars[i].set(f"{raw_deg[i]:.2f}")
                    self.cmd_vars[i].set(f"{cmd[i]:.2f}")
            if self._force_fb_enabled:
                # 主线程刷新力指令缓存，供力反馈线程读取
                self._read_force_cmd_nm(silent=True)
        except Exception:
            pass
        self.root.after(100, self._ui_tick)

    def on_close(self) -> None:
        self.stop_force_feedback()
        self.stop_follow()
        self.disconnect_master()
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        with self._log_lock:
            if self._log_fp is not None and not self._log_fp.closed:
                try:
                    ended = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._log_fp.write(f"=== GUI 退出 {ended} ===\n")
                    self._log_fp.close()
                except Exception:
                    pass
                self._log_fp = None
        self.root.destroy()


def main() -> None:
    cfg = DEFAULT_CFG
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    TeleopApp(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
