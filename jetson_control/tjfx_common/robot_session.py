"""
机器人会话封装 — 连接 / 只读订阅 / 安全释放
==========================================

供 GUI 与脚本共用，避免各处重复 connect/subscribe 逻辑。

注意: 天机 SDK 本地固定绑定 UDP 4730，同一时刻只能有一个进程持有。
面板 / joint_controller / thor_joint_agent 等互斥。
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Marvin SDK 本地 UDP 绑定端口 (见 contrlSDK/Robot.cpp)
SDK_UDP_PORT = 4730


def tip_di_int(v) -> int:
    if v is None:
        return 0
    if isinstance(v, (bytes, bytearray)):
        return int(v[0]) if len(v) else 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def find_sdk_port_holders(port: int = SDK_UDP_PORT) -> List[Tuple[int, str]]:
    """查找占用 SDK UDP 端口的进程 (pid, cmdline)。"""
    holders: List[Tuple[int, str]] = []
    try:
        out = subprocess.check_output(
            ["ss", "-ulnp"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        try:
            out = subprocess.check_output(
                ["lsof", f"-iUDP:{port}", "-n", "-P"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return holders
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                pid = int(parts[1])
                cmd = _pid_cmdline(pid)
                holders.append((pid, cmd))
        return holders

    for line in out.splitlines():
        # ss 格式示例: 0.0.0.0:4730
        if f":{port}" not in line:
            continue
        if "pid=" not in line:
            continue
        try:
            frag = line.split("pid=", 1)[1]
            pid_s = ""
            for ch in frag:
                if ch.isdigit():
                    pid_s += ch
                elif pid_s:
                    break
            if not pid_s:
                continue
            pid = int(pid_s)
            if pid == os.getpid():
                continue
            holders.append((pid, _pid_cmdline(pid)))
        except Exception:
            continue
    # 去重
    seen = set()
    uniq = []
    for pid, cmd in holders:
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append((pid, cmd))
    return uniq


def _pid_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
        return raw or f"pid={pid}"
    except Exception:
        return f"pid={pid}"


def ping_host(ip: str, count: int = 2, timeout_s: float = 1.0) -> Tuple[bool, str]:
    """ICMP ping 控制柜。返回 (ok, 摘要)。"""
    try:
        out = subprocess.check_output(
            ["ping", "-c", str(count), "-W", str(max(1, int(timeout_s))), ip],
            text=True, stderr=subprocess.STDOUT, timeout=count * timeout_s + 2,
        )
        ok = "0% packet loss" in out or " 0% packet" in out
        # 取 rtt 一行
        rtt = ""
        for line in out.splitlines():
            if "rtt" in line or "round-trip" in line:
                rtt = line.strip()
                break
        return ok, (rtt or ("可达" if ok else "丢包"))
    except subprocess.CalledProcessError as e:
        return False, f"ping 失败: {(e.output or '')[-120:]}"
    except Exception as e:
        return False, f"ping 异常: {e}"


def release_sdk_port_holders(
    port: int = SDK_UDP_PORT,
    force: bool = False,
    exclude_pids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """结束占用 SDK UDP 端口的进程。

    先 SIGTERM，仍占用则可选 SIGKILL。
    返回 {ok, killed:[(pid,cmd)], failed:[(pid,err)], remaining:[(pid,cmd)]}
    """
    exclude = set(exclude_pids or [])
    exclude.add(os.getpid())
    holders = [(p, c) for p, c in find_sdk_port_holders(port) if p not in exclude]
    report: Dict[str, Any] = {
        "ok": True, "killed": [], "failed": [], "remaining": [], "port": port,
    }
    if not holders:
        return report

    import signal

    for pid, cmd in holders:
        try:
            os.kill(pid, signal.SIGTERM)
            report["killed"].append((pid, cmd, "SIGTERM"))
        except ProcessLookupError:
            report["killed"].append((pid, cmd, "already-gone"))
        except PermissionError as e:
            report["failed"].append((pid, cmd, f"权限不足: {e}"))
            report["ok"] = False
        except Exception as e:
            report["failed"].append((pid, cmd, str(e)))
            report["ok"] = False

    time.sleep(0.6)
    still = [(p, c) for p, c in find_sdk_port_holders(port) if p not in exclude]
    if still and force:
        for pid, cmd in still:
            try:
                os.kill(pid, signal.SIGKILL)
                report["killed"].append((pid, cmd, "SIGKILL"))
            except ProcessLookupError:
                pass
            except Exception as e:
                report["failed"].append((pid, cmd, str(e)))
                report["ok"] = False
        time.sleep(0.3)
        still = [(p, c) for p, c in find_sdk_port_holders(port) if p not in exclude]

    report["remaining"] = still
    if still:
        report["ok"] = False
    return report


def run_link_diagnosis(ip: str, session: Optional["RobotSession"] = None) -> Dict[str, Any]:
    """通信链路一键诊断 (不强制改模式/不动臂)。"""
    result: Dict[str, Any] = {
        "ip": ip,
        "steps": [],
        "pass": True,
        "holders": [],
    }

    def step(name: str, ok: bool, detail: str):
        result["steps"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            result["pass"] = False

    ping_ok, ping_detail = ping_host(ip)
    step("控制柜 ping", ping_ok, ping_detail)

    holders = find_sdk_port_holders()
    result["holders"] = holders
    if holders:
        detail = "; ".join(f"pid={p} {_short_cmd(c)}" for p, c in holders)
        # 若占用者就是当前 session 自己的进程 — 不应发生，因为 find 排除了 getpid
        step(f"UDP {SDK_UDP_PORT}", False, f"被占用: {detail}")
    else:
        step(f"UDP {SDK_UDP_PORT}", True, "空闲")

    if session is not None and session.connected:
        step("会话状态", True, f"已连接 {session.ip}")
        try:
            ser0 = None
            adv = 0
            for _ in range(6):
                sub = session.subscribe()
                if not sub:
                    time.sleep(0.05)
                    continue
                ser = int(sub["outputs"][0].get("frame_serial", 0) or 0)
                if ser0 is not None and ser != ser0:
                    adv += 1
                ser0 = ser
                time.sleep(0.05)
            step("帧刷新", adv >= 1, f"advance={adv} last_frame={ser0}")
        except Exception as e:
            step("帧刷新", False, str(e))
        try:
            for arm in ("A", "B"):
                codes = session.robot.get_servo_error_code(arm)
                s = str(codes).strip() if codes is not None else "None"
                ok = s in ("None", "None fault codes list", "")
                step(f"伺服 {arm}", ok, s if not ok else "无故障")
        except Exception as e:
            step("伺服诊断", False, str(e))
    else:
        step("会话状态", True, "未连接 (仅检查链路/端口)")
        if not holders and ping_ok:
            # 短连探测: 临时 connect → 读帧 → release，不留给调用方
            probe = RobotSession()
            ok = probe.connect(ip, retries=1)
            if ok:
                step("短连探测", True, "connect+frame OK，已释放")
                probe.close()
            else:
                step("短连探测", False, probe.last_error or "失败")
                probe.close()

    return result


def _short_cmd(cmd: str, n: int = 80) -> str:
    cmd = cmd or ""
    return cmd if len(cmd) <= n else cmd[: n - 1] + "…"


class RobotSession:
    """Concise_Marvin_Robot 会话。默认只读；写操作由调用方显式发起。"""

    def __init__(self):
        self.robot = None
        self.dcss = None
        self.ip: Optional[str] = None
        self.connected = False
        self.last_error: Optional[str] = None
        self.last_warning: Optional[str] = None

    @staticmethod
    def _arm_health(sub: dict, arm: str) -> Tuple[int, int]:
        idx = 0 if str(arm).upper() == "A" else 1
        st = (sub.get("states") or [{}])[idx]
        return (
            int(st.get("err_code", 0) or 0),
            int(st.get("cur_state", 0) or 0),
        )

    def _adopt(self, robot, dcss, ip: str, warning: Optional[str] = None) -> None:
        try:
            robot.local_log_switch("0")
        except Exception:
            pass
        self.robot = robot
        self.dcss = dcss
        self.ip = ip
        self.connected = True
        self.last_error = None
        self.last_warning = warning
        logger.info("RobotSession 已连接 %s%s", ip, f" ({warning})" if warning else "")

    @staticmethod
    def sdk_clear_error(robot, arm: str) -> None:
        """Marvin 有 clear_error；Concise 需走底层 OnClearErr_A/B。"""
        arm = str(arm).upper()
        if hasattr(robot, "clear_error"):
            robot.clear_error(arm)
            return
        raw = getattr(robot, "robot", None)
        fn = getattr(raw, f"OnClearErr_{arm}", None) if raw is not None else None
        if callable(fn):
            fn()
            return
        if hasattr(robot, "servo_reset"):
            for i in range(7):
                robot.servo_reset(arm, i)
            return
        raise AttributeError("无清错接口 (clear_error / OnClearErr)")

    def clear_error(self, arm: str = "A") -> bool:
        if not self.robot:
            return False
        arms = ["A", "B"] if str(arm).upper() == "AB" else [str(arm).upper()]
        ok = True
        for a in arms:
            try:
                self.sdk_clear_error(self.robot, a)
            except Exception as e:
                logger.warning("clear_error %s: %s", a, e)
                ok = False
        return ok

    def _required_arms_ok(self, sub: dict, require_arms: List[str]) -> Tuple[bool, str]:
        bad: List[str] = []
        notes: List[str] = []
        for a in ("A", "B"):
            err, state = self._arm_health(sub, a)
            if err == 0 and state != 100:
                continue
            tag = f"{a}: err={err} state={state}"
            if a in require_arms:
                bad.append(tag)
                notes.append(tag)
            else:
                notes.append(tag + " (忽略)")
        return (not bad), "; ".join(notes)

    @staticmethod
    def _link_onlinkto(robot, ip: str) -> bool:
        """只建 UDP 链路。Concise.Connect 会因任一臂报错返回 False。"""
        ip1, ip2, ip3, ip4 = robot._convert_ip(ip)
        return bool(robot.robot.OnLinkTo(ip1, ip2, ip3, ip4))

    def _try_keep_after_arm_fault(self, robot, ip: str, require_arms: List[str], last_err: str) -> bool:
        """Concise.connect 会因另一臂报错抛异常；目标臂正常则保住会话。"""
        if robot is None:
            return False
        text = last_err or ""
        if "柜可达" not in text and "手臂报错" not in text and "PDO" not in text:
            return False
        from SDK_PYTHON.fx_robot import DCSS

        try:
            dcss = DCSS()
            sub = robot.subscribe(dcss)
            if not sub:
                return False
            ok, note = self._required_arms_ok(sub, require_arms)
            if not ok:
                return False
            warn = text if not note else f"{note}; SDK: {text}"
            self._adopt(robot, dcss, ip, warning=warn)
            return True
        except Exception as e:
            logger.debug("salvage connect failed: %s", e)
            return False

    def connect(
        self,
        ip: str,
        retries: int = 3,
        require_arms: Optional[List[str]] = None,
    ) -> bool:
        from SDK_PYTHON.fx_robot import Concise_Marvin_Robot, DCSS

        require = [a.upper() for a in (require_arms or ["A"])]
        if "AB" in require:
            require = ["A", "B"]

        self.close()
        self.last_error = None
        self.last_warning = None
        holders = find_sdk_port_holders()
        if holders:
            detail = "; ".join(f"pid={p} {c}" for p, c in holders)
            self.last_error = (
                f"SDK UDP {SDK_UDP_PORT} 已被占用 → {detail}。"
                f"请先停止该进程 (常见: thor_joint_agent / 残留 debug_panel)，"
                f"再连接。例: kill {holders[0][0]}"
            )
            logger.error("%s", self.last_error)
            return False

        last_err = None
        for i in range(retries):
            robot = None
            try:
                robot = Concise_Marvin_Robot()
                ok = self._link_onlinkto(robot, ip)
                if not ok:
                    try:
                        robot.release_robot()
                    except Exception:
                        pass
                    holders = find_sdk_port_holders()
                    if holders:
                        detail = "; ".join(f"pid={p} {c}" for p, c in holders)
                        last_err = f"port occupied: {detail}"
                    else:
                        last_err = "OnLinkTo 返回 False (柜无响应或网线异常)"
                    time.sleep(0.3)
                    continue
                dcss = DCSS()
                tag = 0
                last = None
                sub = None
                for _ in range(10):
                    sub = robot.subscribe(dcss)
                    if not sub:
                        time.sleep(0.01)
                        continue
                    ser = sub["outputs"][0].get("frame_serial", 0) or 0
                    if ser and ser != last:
                        tag += 1
                        last = ser
                    time.sleep(0.01)
                if tag == 0:
                    robot.release_robot()
                    last_err = "UDP frame 未刷新 (已连上端口但无柜端数据)"
                    time.sleep(0.3)
                    continue
                warn_parts: List[str] = []
                if sub:
                    _, note = self._required_arms_ok(sub, require)
                    if note:
                        warn_parts.append(note)
                    for a in require:
                        err, state = self._arm_health(sub, a)
                        if err != 0 or state == 100:
                            try:
                                self.sdk_clear_error(robot, a)
                                time.sleep(0.2)
                                warn_parts.append(f"已对{a}下发清错(err={err} state={state})")
                            except Exception as e:
                                warn_parts.append(f"{a}清错失败: {e}")
                self._adopt(robot, dcss, ip, warning="; ".join(warn_parts) or None)
                return True
            except Exception as e:
                last_err = str(e)
                if self._try_keep_after_arm_fault(robot, ip, require, last_err):
                    return True
                if robot is not None:
                    try:
                        robot.release_robot()
                    except Exception:
                        pass
                time.sleep(0.3)
        self.last_error = last_err or "未知错误"
        logger.error("连接失败: %s", self.last_error)
        return False

    def subscribe(self) -> Optional[dict]:
        if not self.connected or self.robot is None:
            return None
        try:
            return self.robot.subscribe(self.dcss)
        except Exception as e:
            logger.debug("subscribe 异常: %s", e)
            return None

    def snapshot_arm(self, sub: dict, arm: str) -> Dict[str, Any]:
        """从 subscribe 结果提取单臂监控字典。"""
        idx = 0 if arm.upper() == "A" else 1
        out = sub["outputs"][idx]
        st = sub["states"][idx]
        inp = sub["inputs"][idx] if idx < len(sub.get("inputs", [])) else {}
        return {
            "arm": arm.upper(),
            # 关节反馈 (控制柜 outputs，上线预检用)
            "joint_pos": list(out.get("fb_joint_pos", [0.0] * 7)),
            "joint_vel": list(out.get("fb_joint_vel", [0.0] * 7)),
            "joint_cmd": list(out.get("fb_joint_cmd", [0.0] * 7)),
            "joint_pos_e": list(out.get("fb_joint_posE", [0.0] * 7)),
            "joint_sns": list(out.get("fb_joint_sToq", [0.0] * 7)),
            "joint_cmd_trq": list(out.get("fb_joint_cToq", [0.0] * 7)),
            "joint_temp": list(out.get("fb_joint_them", [0.0] * 7)),
            "joint_fric": list(out.get("est_joint_firc", [0.0] * 7)),
            "joint_ext": list(out.get("est_joint_force", [0.0] * 7)),
            "cart_ext": list(out.get("est_cart_fn", [0.0] * 6)),
            "frame_serial": int(out.get("frame_serial", 0) or 0),
            "tip_di": tip_di_int(out.get("tip_di", 0)),
            "cur_state": int(st.get("cur_state", 0) or 0),
            "cmd_state": int(st.get("cmd_state", 0) or 0),
            "err_code": int(st.get("err_code", 0) or 0),
            "imp_type": int(inp.get("imp_type", 0) or 0),
            "force_cmd": float(inp.get("force_cmd", 0) or 0),
            "force_type": int(inp.get("force_type", 0) or 0),
            "force_dir": list(inp.get("force_dir", [0.0] * 6)),
            "force_adj_lmt": float(inp.get("force_adj_lmt", 0) or 0),
            "joint_k": list(inp.get("joint_k", [0.0] * 7)),
            "joint_d": list(inp.get("joint_d", [0.0] * 7)),
            "cart_k": list(inp.get("cart_k", [0.0] * 6)),
            "cart_d": list(inp.get("cart_d", [0.0] * 6)),
        }

    def soft_stop(self, arm: str = "AB") -> None:
        if not self.robot:
            return
        arms = ["A", "B"] if arm == "AB" else [arm]
        for a in arms:
            try:
                self.robot.soft_stop(a)
            except Exception as e:
                logger.warning("soft_stop %s: %s", a, e)

    def disable(self, arm: str = "AB") -> None:
        if not self.robot:
            return
        arms = ["A", "B"] if arm == "AB" else [arm]
        for a in arms:
            try:
                self.robot.disable(a)
            except Exception:
                pass

    def close(self, disable_first: bool = False) -> None:
        """释放 SDK 端口。GUI 断开时默认不先 disable，尽快 OnRelease 让出 UDP 4730。"""
        if self.robot is None:
            self.connected = False
            return
        if disable_first:
            try:
                self.disable("AB")
            except Exception:
                pass
        try:
            self.robot.release_robot()
        except Exception:
            pass
        try:
            del self.robot
        except Exception:
            pass
        self.robot = None
        self.dcss = None
        self.connected = False
        logger.info("RobotSession 已释放")
