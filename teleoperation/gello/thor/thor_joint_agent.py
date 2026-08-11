#!/usr/bin/env python3
"""
Thor 双臂关节角度控制代理
========================
在 tegra (Thor) 上运行，本机 GUI 通过 TCP JSON 下发双臂角度指令。
SDK / eth1 → 192.168.1.190 只能从 Thor 访问，因此控制环必须在本机代理内。

运行 (Thor):
  cd ~/Desktop/lambda2_jetson_control/jetson_control
  python3 scripts/thor_joint_agent.py --port 15666

协议: 一行一个 JSON 请求/响应 (UTF-8, \\n 结尾)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 允许放在 jetson_control/scripts 或任意目录
CANDIDATE_ROOTS = [
    os.path.dirname(SCRIPT_DIR),  # .../jetson_control
    os.path.expanduser("~/Desktop/lambda2_jetson_control/jetson_control"),
    os.path.expanduser("~/lambda2"),
]
PROJECT_ROOT = None
for root in CANDIDATE_ROOTS:
    if os.path.isdir(os.path.join(root, "tjfx_common")) or os.path.isdir(
        os.path.join(os.path.dirname(root), "sdk")
    ):
        PROJECT_ROOT = root
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.path.expanduser("~/Desktop/lambda2_jetson_control/jetson_control")

SDK_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "sdk", "TJ_FX_ROBOT_CONTRL_SDK-master")
SDK_PYTHON = os.path.join(SDK_DIR, "SDK_PYTHON")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SDK_DIR)
sys.path.insert(0, SDK_PYTHON)

DEFAULT_IP = "192.168.1.190"
DEFAULT_PORT = 15666
JOINT_LIMITS = [
    (-170.0, 170.0),
    (-120.0, 120.0),
    (-90.0, 90.0),
    (-170.0, 170.0),
    (-120.0, 120.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
]
MAX_STEP_DEG = 8.0  # 单次指令相对上一目标的最大步长
HARD_MAX_VEL = 40
MAX_CART_STEP_MM = 8.0  # 单次笛卡尔平移上限 (mm) — 安全
MAX_CART_STEP_DEG = 3.0  # 单次姿态增量上限 (deg)
KINE_CFG_NAME = "ccs_m6_40.MvKDCfg"


DEFAULT_JOINT_K = [5.0, 5.0, 5.0, 4.0, 3.0, 3.0, 2.0]
DEFAULT_JOINT_D = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]


class DualArmAgent:
    def __init__(self):
        self._lock = threading.RLock()
        self.session = None
        self.robot_ip = DEFAULT_IP
        self.enabled: Dict[str, bool] = {"A": False, "B": False}
        self.target: Dict[str, List[float]] = {
            "A": [0.0] * 7,
            "B": [0.0] * 7,
        }
        self.target_xyzabc: Dict[str, List[float]] = {
            "A": [0.0] * 6,
            "B": [0.0] * 6,
        }
        self.vel_pct = 20
        self.acc_pct = 20
        # 与上位机统一的控制参数
        self.control_hz = 100.0
        self.control_mode = "impedance"  # position | impedance
        self.max_step_deg = float(MAX_STEP_DEG)
        self.joint_k = list(DEFAULT_JOINT_K)
        self.joint_d = list(DEFAULT_JOINT_D)
        self._kine = {"A": None, "B": None}
        self._kine_ready = False
        self._kine_msg = "未初始化"

    def _ensure_session(self):
        from tjfx_common.robot_session import RobotSession

        if self.session is None:
            self.session = RobotSession()

    def connect(self, ip: str) -> Dict[str, Any]:
        with self._lock:
            self._ensure_session()
            ok = self.session.connect(ip)
            self.robot_ip = ip
            self.enabled = {"A": False, "B": False}
            if not ok:
                return {"ok": False, "msg": f"连接失败: {ip}"}
            snap = self._snapshot()
            for arm in ("A", "B"):
                self.target[arm] = list(snap["arms"][arm]["joint_pos"])
            kine_ok = self._init_kine()
            for arm in ("A", "B"):
                self._sync_xyzabc_from_joints(arm)
            return {
                "ok": True,
                "msg": f"已连接 {ip}; kine={'OK' if kine_ok else self._kine_msg}",
                "kine_ready": self._kine_ready,
                "state": snap,
            }

    def _init_kine(self) -> bool:
        """加载双臂运动学，供笛卡尔增量 IK。失败时 cart_delta 不可用。"""
        try:
            import logging

            logging.disable(logging.INFO)
            from fx_kine import Marvin_Kine

            common = os.path.join(SDK_DIR, "CommonConfig")
            cfg_path = os.path.join(common, KINE_CFG_NAME)
            if not os.path.isfile(cfg_path):
                self._kine_ready = False
                self._kine_msg = f"缺少运动学配置: {cfg_path}"
                print(f"[kine] {self._kine_msg}", flush=True)
                return False

            ok_all = True
            for idx, arm in enumerate(("A", "B")):
                kine = Marvin_Kine()
                kine.log_switch(0)
                ini = kine.load_config(arm_type=idx, config_path=cfg_path)
                if not ini:
                    ok_all = False
                    self._kine[arm] = None
                    continue
                ok = kine.initial_kine(
                    robot_type=ini["TYPE"][idx],
                    dh=ini["DH"][idx],
                    pnva=ini["PNVA"][idx],
                    j67=ini["BD"][idx],
                )
                if not ok:
                    ok_all = False
                    self._kine[arm] = None
                else:
                    self._kine[arm] = kine
                    print(f"[kine] {arm} OK TYPE={ini['TYPE'][idx]}", flush=True)
            self._kine_ready = ok_all and all(self._kine[a] is not None for a in ("A", "B"))
            self._kine_msg = "OK" if self._kine_ready else "部分/全部运动学初始化失败"
            return self._kine_ready
        except Exception as e:
            self._kine_ready = False
            self._kine_msg = f"{type(e).__name__}: {e}"
            print(f"[kine] FAIL {self._kine_msg}", flush=True)
            return False

    def _fk_xyzabc(self, arm: str, joints: List[float]) -> Optional[List[float]]:
        kine = self._kine.get(arm)
        if kine is None:
            return None
        mat = kine.fk(list(joints))
        if mat is False:
            return None
        return list(kine.mat4x4_to_xyzabc(mat))

    def _ik_joints(self, arm: str, xyzabc: List[float], ref_joints: List[float]) -> Optional[List[float]]:
        from fx_kine import FX_InvKineSolvePara

        kine = self._kine.get(arm)
        if kine is None:
            return None
        mat = kine.xyzabc_to_mat4x4(list(xyzabc))
        if mat is False:
            return None
        mat16 = kine.mat4x4_to_mat1x16(mat)
        sp = FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(mat16)
        sp.set_input_ik_ref_joint(list(ref_joints))
        sp.set_input_ik_zsp_type(0)
        ret = kine.ik(sp)
        if not ret:
            return None
        sp = ret if hasattr(ret, "m_Output_RetJoint") else sp
        if getattr(sp, "m_Output_IsOutRange", 0):
            return None
        if getattr(sp, "m_Output_IsJntExd", 0):
            return None
        if hasattr(sp, "get_output_ret_joint"):
            joints = sp.get_output_ret_joint()
        else:
            joints = sp.m_Output_RetJoint.to_list()
        if joints is None:
            return None
        return list(joints)

    def _sync_xyzabc_from_joints(self, arm: str) -> None:
        xyz = self._fk_xyzabc(arm, self.target[arm])
        if xyz is not None:
            self.target_xyzabc[arm] = xyz

    def disconnect(self) -> Dict[str, Any]:
        with self._lock:
            if self.session and self.session.connected:
                try:
                    for arm in ("A", "B"):
                        if self.enabled.get(arm):
                            self._disable_arm(arm)
                except Exception:
                    pass
                self.session.close()
            self.enabled = {"A": False, "B": False}
            return {"ok": True, "msg": "已断开"}

    def _snapshot(self) -> Dict[str, Any]:
        if not self.session or not self.session.connected:
            return {
                "connected": False,
                "ip": self.robot_ip,
                "arms": {
                    "A": self._empty_arm("A"),
                    "B": self._empty_arm("B"),
                },
            }
        sub = self.session.subscribe()
        if not sub:
            return {
                "connected": True,
                "ip": self.robot_ip,
                "arms": {
                    "A": self._empty_arm("A"),
                    "B": self._empty_arm("B"),
                },
                "msg": "subscribe 失败",
            }
        arms = {}
        for arm in ("A", "B"):
            d = self.session.snapshot_arm(sub, arm)
            pe = d.get("joint_pos_e") or d["joint_pos"]
            out = dict(d)
            # JSON 友好：列表化 + 圆整
            for k, n in (
                ("joint_pos", 7), ("joint_vel", 7), ("joint_cmd", 7),
                ("joint_pos_e", 7), ("joint_sns", 7), ("joint_cmd_trq", 7),
                ("joint_temp", 7), ("joint_fric", 7), ("joint_ext", 7),
                ("cart_ext", 6), ("force_dir", 6),
                ("joint_k", 7), ("joint_d", 7), ("cart_k", 6), ("cart_d", 6),
            ):
                vals = out.get(k) or ([0.0] * n)
                out[k] = [round(float(x), 4) for x in list(vals)[:n]]
            out["joint_pos_e"] = [round(float(x), 4) for x in pe]
            out["enabled"] = bool(self.enabled.get(arm))
            out["target"] = [round(float(x), 3) for x in self.target[arm]]
            arms[arm] = out
        return {"connected": True, "ip": self.robot_ip, "arms": arms}

    @staticmethod
    def _empty_arm(arm: str) -> Dict[str, Any]:
        z7 = [0.0] * 7
        z6 = [0.0] * 6
        return {
            "arm": arm,
            "joint_pos": list(z7),
            "joint_pos_e": list(z7),
            "joint_vel": list(z7),
            "joint_cmd": list(z7),
            "joint_sns": list(z7),
            "joint_cmd_trq": list(z7),
            "joint_temp": list(z7),
            "joint_fric": list(z7),
            "joint_ext": list(z7),
            "cart_ext": list(z6),
            "frame_serial": 0,
            "tip_di": 0,
            "cur_state": -1,
            "cmd_state": -1,
            "err_code": -1,
            "imp_type": 0,
            "force_cmd": 0.0,
            "force_type": 0,
            "force_dir": list(z6),
            "force_adj_lmt": 0.0,
            "joint_k": list(z7),
            "joint_d": list(z7),
            "cart_k": list(z6),
            "cart_d": list(z6),
            "enabled": False,
            "target": list(z7),
        }

    def rpc(self, method: str, args: Optional[List[Any]] = None) -> Dict[str, Any]:
        """转发 Concise_Marvin_Robot 方法（供本机 debug_panel 使用）。不自动使能。"""
        args = list(args or [])
        with self._lock:
            if not self.session or not self.session.connected:
                return {"ok": False, "msg": "未连接"}
            r = self.session.robot
            if not hasattr(r, method):
                return {"ok": False, "msg": f"无方法: {method}"}
            fn = getattr(r, method)
            try:
                result = fn(*args)
            except Exception as e:
                return {"ok": False, "msg": f"{method}: {e}", "trace": traceback.format_exc()[-600:]}
            # 标记使能状态（仅当明确进入控制态）
            if method in ("set_position_state", "set_imp_joint_state",
                          "set_imp_cart_state", "set_imp_force_state"):
                arm = str(args[0]).upper() if args else "A"
                if arm in ("A", "B"):
                    self.enabled[arm] = True
            if method == "disable":
                arm = str(args[0]).upper() if args else "A"
                if arm in ("A", "B"):
                    self.enabled[arm] = False
                elif arm == "AB":
                    self.enabled = {"A": False, "B": False}
            if method == "soft_stop":
                arm = str(args[0]).upper() if args else "AB"
                for a in (["A", "B"] if arm == "AB" else [arm]):
                    if a in self.enabled:
                        self.enabled[a] = False
            # JSON 序列化
            if isinstance(result, (bytes, bytearray)):
                result = result.decode("utf-8", errors="replace")
            return {"ok": True, "result": result, "method": method}

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"ok": True, "state": self._snapshot()}

    @staticmethod
    def _normalize_mode(mode: Optional[str]) -> str:
        m = (mode or "position").strip().lower()
        if m in ("impedance", "joint_impedance", "imp_joint", "joint_imp"):
            return "impedance"
        return "position"

    def set_params(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """上位机统一下发控制频率 / 模式 / 步长 / 阻抗参数。"""
        with self._lock:
            if req.get("frequency_hz") is not None:
                self.control_hz = max(1.0, min(500.0, float(req["frequency_hz"])))
            if req.get("control_mode") is not None:
                self.control_mode = self._normalize_mode(str(req["control_mode"]))
            if req.get("max_step_deg") is not None:
                self.max_step_deg = max(0.1, min(30.0, float(req["max_step_deg"])))
            if req.get("vel_ratio") is not None:
                self.vel_pct = max(1, min(int(req["vel_ratio"]), HARD_MAX_VEL))
            if req.get("acc_ratio") is not None:
                self.acc_pct = max(1, min(int(req["acc_ratio"]), HARD_MAX_VEL))
            if req.get("joint_k") is not None:
                k = [float(x) for x in list(req["joint_k"])[:7]]
                while len(k) < 7:
                    k.append(DEFAULT_JOINT_K[len(k)])
                self.joint_k = k
            if req.get("joint_d") is not None:
                d = [float(x) for x in list(req["joint_d"])[:7]]
                while len(d) < 7:
                    d.append(DEFAULT_JOINT_D[len(d)])
                self.joint_d = d
            return {
                "ok": True,
                "msg": "params updated",
                "frequency_hz": self.control_hz,
                "control_mode": self.control_mode,
                "max_step_deg": self.max_step_deg,
                "vel_ratio": self.vel_pct,
                "acc_ratio": self.acc_pct,
                "joint_k": list(self.joint_k),
                "joint_d": list(self.joint_d),
            }

    def enable(
        self,
        arm: str,
        vel: int = 20,
        acc: int = 20,
        mode: Optional[str] = None,
        K: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """进入控制模式：position(cur_state=1) 或 impedance(cur_state=3,imp_type=1)。"""
        arm = arm.upper()
        arms = ["A", "B"] if arm == "AB" else [arm]
        mode_n = self._normalize_mode(mode if mode is not None else self.control_mode)
        with self._lock:
            if not self.session or not self.session.connected:
                return {"ok": False, "msg": "未连接"}
            vel = max(1, min(int(vel), HARD_MAX_VEL))
            acc = max(1, min(int(acc), HARD_MAX_VEL))
            self.vel_pct, self.acc_pct = vel, acc
            self.control_mode = mode_n
            if K is not None:
                self.joint_k = [float(x) for x in list(K)[:7]]
                while len(self.joint_k) < 7:
                    self.joint_k.append(DEFAULT_JOINT_K[len(self.joint_k)])
            if D is not None:
                self.joint_d = [float(x) for x in list(D)[:7]]
                while len(self.joint_d) < 7:
                    self.joint_d.append(DEFAULT_JOINT_D[len(self.joint_d)])
            results = {}
            for a in arms:
                try:
                    if mode_n == "impedance":
                        results[a] = self._enter_impedance(
                            a, vel, acc, self.joint_k, self.joint_d
                        )
                    else:
                        results[a] = self._enter_position(a, vel, acc)
                except Exception as e:
                    results[a] = {
                        "ok": False,
                        "msg": str(e),
                        "trace": traceback.format_exc()[-400:],
                    }
            all_ok = all(v.get("ok") for v in results.values())
            label = "关节阻抗" if mode_n == "impedance" else "位置跟随"
            return {
                "ok": all_ok,
                "msg": f"{label} enable {arms} vel={vel}%",
                "control_mode": mode_n,
                "results": results,
                "state": self._snapshot(),
            }

    def _arm_cur_state(self, arm: str) -> int:
        snap = self._snapshot()
        return int(snap["arms"][arm].get("cur_state", -1))

    def _arm_err(self, arm: str) -> int:
        snap = self._snapshot()
        return int(snap["arms"][arm].get("err_code", -1))

    def _wait_state(
        self, arm: str, expect: int, timeout_s: float = 3.0, allow_transient: bool = True
    ) -> int:
        """等待 cur_state 到达 expect；101/102/103 为切换中。"""
        transient = {101, 102, 103}
        t0 = time.time()
        last = self._arm_cur_state(arm)
        while time.time() - t0 < timeout_s:
            last = self._arm_cur_state(arm)
            if last == expect:
                return last
            if last == 100:
                return last  # 伺服报错，别空等
            if allow_transient and last in transient:
                time.sleep(0.08)
                continue
            time.sleep(0.08)
        return last

    def _prep_enter_mode(self, arm: str, err0: int) -> None:
        r = self.session.robot
        try:
            if err0 != 0:
                r.clear_error(arm)
                time.sleep(0.25)
        except Exception:
            pass
        cur = self._arm_cur_state(arm)
        if cur not in (0,):
            try:
                r.disable(arm)
            except Exception:
                pass
            self._wait_state(arm, 0, timeout_s=2.0, allow_transient=True)

    def _mark_enabled(self, arm: str, ready: bool) -> Dict[str, Any]:
        self.enabled[arm] = ready
        snap = self._snapshot()
        self.target[arm] = list(snap["arms"][arm]["joint_pos"])
        self._sync_xyzabc_from_joints(arm)
        return snap

    def _enter_position(self, arm: str, vel: int, acc: int) -> Dict[str, Any]:
        r = self.session.robot

        # 0) 已在位置跟随则直接就绪，避免 disable↔enable 卡死占锁
        cur0 = self._arm_cur_state(arm)
        err0 = self._arm_err(arm)
        if cur0 == 1 and err0 == 0:
            try:
                r.set_position_state(arm, vel, acc)
            except Exception:
                pass
            snap = self._mark_enabled(arm, True)
            return {
                "ok": True,
                "api_ok": True,
                "cur_state": 1,
                "err_code": 0,
                "imp_type": int(snap["arms"][arm].get("imp_type", 0)),
                "msg": "已在位置跟随(state=1)，直接就绪",
            }

        self._prep_enter_mode(arm, err0)

        ok = bool(r.set_position_state(arm, vel, acc))
        time.sleep(0.15)
        cur = self._wait_state(arm, 1, timeout_s=3.0, allow_transient=True)

        if cur == 101:
            ok = bool(r.set_position_state(arm, vel, acc)) or ok
            cur = self._wait_state(arm, 1, timeout_s=2.5, allow_transient=True)

        err = self._arm_err(arm)
        ready = cur == 1 and err == 0
        snap = self._mark_enabled(arm, ready)
        return {
            "ok": ready,
            "api_ok": ok,
            "cur_state": cur,
            "err_code": err,
            "imp_type": int(snap["arms"][arm].get("imp_type", 0)),
            "msg": (
                "已进入位置跟随(state=1)"
                if ready
                else f"未到位: cur_state={cur}(期望1/位置) err={err} api={ok}"
            ),
        }

    def _enter_impedance(
        self,
        arm: str,
        vel: int,
        acc: int,
        K: List[float],
        D: List[float],
    ) -> Dict[str, Any]:
        """进入关节阻抗: cur_state=3, imp_type=1。"""
        r = self.session.robot
        K = [float(x) for x in list(K)[:7]]
        D = [float(x) for x in list(D)[:7]]
        while len(K) < 7:
            K.append(DEFAULT_JOINT_K[len(K)])
        while len(D) < 7:
            D.append(DEFAULT_JOINT_D[len(D)])

        cur0 = self._arm_cur_state(arm)
        err0 = self._arm_err(arm)
        snap0 = self._snapshot()
        imp0 = int(snap0["arms"][arm].get("imp_type", 0))
        if cur0 == 3 and err0 == 0 and imp0 == 1:
            try:
                r.set_imp_joint_state(arm, vel, acc, K, D)
            except Exception:
                pass
            snap = self._mark_enabled(arm, True)
            return {
                "ok": True,
                "api_ok": True,
                "cur_state": 3,
                "err_code": 0,
                "imp_type": 1,
                "msg": "已在关节阻抗(state=3,imp=1)，直接就绪",
                "joint_k": list(K),
                "joint_d": list(D),
            }

        self._prep_enter_mode(arm, err0)

        ok = bool(r.set_imp_joint_state(arm, vel, acc, K, D))
        time.sleep(0.2)
        cur = self._wait_state(arm, 3, timeout_s=3.5, allow_transient=True)
        if cur == 103:
            ok = bool(r.set_imp_joint_state(arm, vel, acc, K, D)) or ok
            cur = self._wait_state(arm, 3, timeout_s=2.5, allow_transient=True)

        err = self._arm_err(arm)
        snap = self._snapshot()
        imp = int(snap["arms"][arm].get("imp_type", 0))
        ready = cur == 3 and err == 0 and imp == 1
        if not ready and cur == 3 and err == 0:
            # 部分固件回读 imp_type 稍慢，扭矩态也先放行
            ready = True
        self._mark_enabled(arm, ready)
        return {
            "ok": ready,
            "api_ok": ok,
            "cur_state": cur,
            "err_code": err,
            "imp_type": imp,
            "msg": (
                "已进入关节阻抗(state=3,imp=1)"
                if ready
                else f"未到位: cur_state={cur}(期望3) imp={imp} err={err} api={ok}"
            ),
            "joint_k": list(K),
            "joint_d": list(D),
        }

    def _disable_arm(self, arm: str):
        if self.session and self.session.robot:
            try:
                self.session.robot.disable(arm)
            except Exception:
                try:
                    self.session.robot.set_state(arm, 0)
                except Exception:
                    pass
            self._wait_state(arm, 0, timeout_s=2.0, allow_transient=True)
        self.enabled[arm] = False

    def disable(self, arm: str = "AB") -> Dict[str, Any]:
        arm = arm.upper()
        arms = ["A", "B"] if arm == "AB" else [arm]
        with self._lock:
            for a in arms:
                self._disable_arm(a)
            return {"ok": True, "msg": f"已下使能 {arms}", "state": self._snapshot()}

    def soft_stop(self, arm: str = "AB") -> Dict[str, Any]:
        with self._lock:
            if self.session:
                self.session.soft_stop(arm)
            for a in (["A", "B"] if arm.upper() == "AB" else [arm.upper()]):
                self.enabled[a] = False
            return {"ok": True, "msg": "软急停", "state": self._snapshot()}

    def clear_error(self, arm: str = "AB") -> Dict[str, Any]:
        arm = arm.upper()
        arms = ["A", "B"] if arm == "AB" else [arm]
        with self._lock:
            if not self.session or not self.session.connected:
                return {"ok": False, "msg": "未连接"}
            r = self.session.robot
            for a in arms:
                try:
                    r.clear_error(a)
                except Exception:
                    pass
            time.sleep(0.2)
            return {"ok": True, "msg": f"已清错 {arms}", "state": self._snapshot()}

    def _clamp_joints(self, joints: List[float]) -> List[float]:
        out = []
        for i, v in enumerate(joints[:7]):
            lo, hi = JOINT_LIMITS[i]
            out.append(max(lo, min(hi, float(v))))
        while len(out) < 7:
            out.append(0.0)
        return out

    def _limit_step(self, arm: str, joints: List[float]) -> List[float]:
        prev = self.target[arm]
        step = float(self.max_step_deg)
        limited = []
        for i in range(7):
            d = joints[i] - prev[i]
            if d > step:
                d = step
            elif d < -step:
                d = -step
            limited.append(prev[i] + d)
        return limited

    def set_joints(
        self,
        arm: str,
        joints: List[float],
        interp_s: float = 0.0,
        bypass_enable: bool = False,
    ) -> Dict[str, Any]:
        arm = arm.upper()
        if arm not in ("A", "B"):
            return {"ok": False, "msg": "arm 须为 A/B"}
        with self._lock:
            if not self.session or not self.session.connected:
                return {"ok": False, "msg": "未连接"}
            if not self.enabled.get(arm) and not bypass_enable:
                return {"ok": False, "msg": f"臂 {arm} 未使能，先 enable"}

            joints = self._clamp_joints(joints)
            if interp_s and interp_s > 0.05:
                return self._interp_move(arm, joints, float(interp_s))

            joints = self._limit_step(arm, joints)
            ok = self.session.robot.set_joint_position_cmd(arm, joints)
            if ok:
                self.target[arm] = list(joints)
            return {
                "ok": bool(ok),
                "msg": "set_joints" if ok else "set_joint_position_cmd 失败",
                "target": [round(x, 3) for x in self.target[arm]],
            }

    def _interp_move(self, arm: str, target: List[float], duration_s: float) -> Dict[str, Any]:
        start = list(self.target[arm])
        hz = max(1.0, float(self.control_hz))
        n = max(1, int(duration_s * hz))
        period = 1.0 / hz
        r = self.session.robot
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            cmd = [start[i] + (target[i] - start[i]) * a for i in range(7)]
            if not r.set_joint_position_cmd(arm, cmd):
                return {"ok": False, "msg": f"插值失败 @frame {k}"}
            self.target[arm] = cmd
            time.sleep(period)
        for _ in range(5):
            r.set_joint_position_cmd(arm, target)
            time.sleep(period)
        self.target[arm] = list(target)
        return {
            "ok": True,
            "msg": f"插值完成 {duration_s:.1f}s",
            "target": [round(x, 3) for x in target],
            "state": self._snapshot(),
        }

    def sync_target(self, arm: str = "AB") -> Dict[str, Any]:
        arm = arm.upper()
        arms = ["A", "B"] if arm == "AB" else [arm]
        with self._lock:
            snap = self._snapshot()
            for a in arms:
                self.target[a] = list(snap["arms"][a]["joint_pos"])
                self._sync_xyzabc_from_joints(a)
            return {
                "ok": True,
                "msg": "目标已同步到反馈",
                "kine_ready": self._kine_ready,
                "xyzabc": {
                    a: [round(x, 2) for x in self.target_xyzabc[a]] for a in arms
                },
                "state": snap,
            }

    def cart_delta(
        self,
        arm: str,
        dxyz_mm: Optional[List[float]] = None,
        drpy_deg: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        位置模式下末端相对增量 (mm/deg) → IK → set_joint_position_cmd。
        单步硬限幅；IK 失败则不同步目标、不发指令。
        """
        arm = arm.upper()
        if arm not in ("A", "B"):
            return {"ok": False, "msg": "arm 须为 A/B"}
        dxyz_mm = list(dxyz_mm or [0.0, 0.0, 0.0])
        drpy_deg = list(drpy_deg or [0.0, 0.0, 0.0])
        while len(dxyz_mm) < 3:
            dxyz_mm.append(0.0)
        while len(drpy_deg) < 3:
            drpy_deg.append(0.0)

        with self._lock:
            if not self.session or not self.session.connected:
                return {"ok": False, "msg": "未连接"}
            if not self.enabled.get(arm):
                return {"ok": False, "msg": f"臂 {arm} 未使能"}
            if not self._kine_ready or self._kine.get(arm) is None:
                return {"ok": False, "msg": f"运动学未就绪: {self._kine_msg}"}

            # 限幅
            for i in range(3):
                dxyz_mm[i] = max(-MAX_CART_STEP_MM, min(MAX_CART_STEP_MM, float(dxyz_mm[i])))
                drpy_deg[i] = max(-MAX_CART_STEP_DEG, min(MAX_CART_STEP_DEG, float(drpy_deg[i])))

            if all(abs(v) < 1e-6 for v in dxyz_mm[:3] + drpy_deg[:3]):
                return {
                    "ok": True,
                    "msg": "零增量",
                    "xyzabc": [round(x, 2) for x in self.target_xyzabc[arm]],
                    "target": [round(x, 3) for x in self.target[arm]],
                }

            new_xyz = list(self.target_xyzabc[arm])
            for i in range(3):
                new_xyz[i] += dxyz_mm[i]
                new_xyz[i + 3] += drpy_deg[i]

            joints = self._ik_joints(arm, new_xyz, self.target[arm])
            if joints is None:
                # 回同步，防止目标漂移
                self._sync_xyzabc_from_joints(arm)
                return {
                    "ok": False,
                    "msg": "IK 失败，已同步反馈",
                    "xyzabc": [round(x, 2) for x in self.target_xyzabc[arm]],
                }

            joints = self._clamp_joints(joints)
            joints = self._limit_step(arm, joints)
            ok = self.session.robot.set_joint_position_cmd(arm, joints)
            if not ok:
                return {"ok": False, "msg": "set_joint_position_cmd 失败"}

            self.target[arm] = list(joints)
            # 用实际发出的关节再 FK，保持一致
            self._sync_xyzabc_from_joints(arm)
            return {
                "ok": True,
                "msg": "cart_delta",
                "xyzabc": [round(x, 2) for x in self.target_xyzabc[arm]],
                "target": [round(x, 3) for x in self.target[arm]],
                "applied_dxyz_mm": [round(x, 3) for x in dxyz_mm[:3]],
            }

    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = (req.get("cmd") or "").strip().lower()
        try:
            if cmd == "ping":
                return {
                    "ok": True,
                    "msg": "pong",
                    "ts": time.time(),
                    "kine_ready": self._kine_ready,
                    "frequency_hz": self.control_hz,
                    "control_mode": self.control_mode,
                    "max_step_deg": self.max_step_deg,
                }
            if cmd == "connect":
                return self.connect(req.get("ip") or DEFAULT_IP)
            if cmd == "disconnect":
                return self.disconnect()
            if cmd == "get_state":
                return self.get_state()
            if cmd == "set_params":
                return self.set_params(req)
            if cmd == "enable":
                return self.enable(
                    req.get("arm", "AB"),
                    int(req.get("vel", self.vel_pct)),
                    int(req.get("acc", self.acc_pct)),
                    req.get("mode"),
                    req.get("K") or req.get("joint_k"),
                    req.get("D") or req.get("joint_d"),
                )
            if cmd == "disable":
                return self.disable(req.get("arm", "AB"))
            if cmd == "soft_stop":
                return self.soft_stop(req.get("arm", "AB"))
            if cmd == "clear_error":
                return self.clear_error(req.get("arm", "AB"))
            if cmd == "set_joints":
                return self.set_joints(
                    req.get("arm", "A"),
                    list(req.get("joints") or []),
                    float(req.get("interp_s", 0.0) or 0.0),
                )
            if cmd == "sync_target":
                return self.sync_target(req.get("arm", "AB"))
            if cmd == "cart_delta":
                return self.cart_delta(
                    req.get("arm", "A"),
                    req.get("dxyz_mm"),
                    req.get("drpy_deg"),
                )
            if cmd == "rpc":
                return self.rpc(req.get("method") or "", req.get("args") or [])
            return {"ok": False, "msg": f"未知命令: {cmd}"}
        except Exception as e:
            return {
                "ok": False,
                "msg": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-800:],
            }


def _recv_line(conn: socket.socket, buf: bytearray) -> Optional[str]:
    while True:
        idx = buf.find(b"\n")
        if idx >= 0:
            line = bytes(buf[:idx]).decode("utf-8", errors="replace")
            del buf[: idx + 1]
            return line
        chunk = conn.recv(4096)
        if not chunk:
            return None
        buf.extend(chunk)


def client_loop(conn: socket.socket, addr, agent: DualArmAgent):
    print(f"[client] + {addr}", flush=True)
    buf = bytearray()
    try:
        conn.settimeout(120.0)
        while True:
            line = _recv_line(conn, buf)
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {"ok": False, "msg": f"JSON 错误: {e}"}
            else:
                resp = agent.handle(req)
            data = (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
            conn.sendall(data)
    except Exception as e:
        print(f"[client] err {addr}: {e}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[client] - {addr}", flush=True)


def serve(host: str, port: int):
    agent = DualArmAgent()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    print(f"[agent] listening {host}:{port}", flush=True)
    print(f"[agent] PROJECT_ROOT={PROJECT_ROOT}", flush=True)
    print(f"[agent] SDK_DIR={SDK_DIR}", flush=True)
    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(
                target=client_loop, args=(conn, addr, agent), daemon=True
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[agent] stop", flush=True)
    finally:
        try:
            agent.disconnect()
        except Exception:
            pass
        sock.close()


def main():
    p = argparse.ArgumentParser(description="Thor 双臂关节角度 TCP 代理")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
