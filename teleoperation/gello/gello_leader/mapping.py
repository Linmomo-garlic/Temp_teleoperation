"""Gello Dynamixel ↔ Tianji joint angle mapping."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# Tianji soft limits (deg) — aligned with thor_joint_agent / local_dual_arm_gui
JOINT_LIMITS_DEG: List[Tuple[float, float]] = [
    (-170.0, 170.0),
    (-120.0, 120.0),
    (-90.0, 90.0),
    (-170.0, 170.0),
    (-120.0, 120.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
]


def apply_calibration(
    q_raw_rad: np.ndarray,
    offsets_rad: Sequence[float],
    signs: Sequence[float],
) -> np.ndarray:
    """Master raw → Tianji cmd: q_cmd = (q_raw - offset) * sign, radians."""
    q = np.asarray(q_raw_rad, dtype=float).reshape(-1)
    off = np.asarray(offsets_rad, dtype=float).reshape(-1)
    sgn = np.asarray(signs, dtype=float).reshape(-1)
    if q.size != off.size or q.size != sgn.size:
        raise ValueError("raw / offsets / signs length mismatch")
    return (q - off) * sgn


def invert_calibration(
    q_cmd_rad: np.ndarray,
    offsets_rad: Sequence[float],
    signs: Sequence[float],
) -> np.ndarray:
    """Tianji cmd → Master raw: q_raw = q_cmd / sign + offset, radians."""
    q = np.asarray(q_cmd_rad, dtype=float).reshape(-1)
    off = np.asarray(offsets_rad, dtype=float).reshape(-1)
    sgn = np.asarray(signs, dtype=float).reshape(-1)
    if q.size != off.size or q.size != sgn.size:
        raise ValueError("cmd / offsets / signs length mismatch")
    if np.any(np.abs(sgn) < 1e-9):
        raise ValueError("joint_signs must be non-zero")
    return q / sgn + off


def to_tianji_deg(
    q_cmd_rad: np.ndarray,
    limits: Sequence[Tuple[float, float]] = JOINT_LIMITS_DEG,
) -> List[float]:
    deg = np.rad2deg(np.asarray(q_cmd_rad, dtype=float).reshape(-1))
    out: List[float] = []
    for i, v in enumerate(deg.tolist()):
        lo, hi = limits[i] if i < len(limits) else (-180.0, 180.0)
        out.append(float(max(lo, min(hi, v))))
    while len(out) < 7:
        out.append(0.0)
    return out[:7]


def clamp_tianji_deg(
    deg: Sequence[float],
    limits: Sequence[Tuple[float, float]] = JOINT_LIMITS_DEG,
) -> List[float]:
    out: List[float] = []
    for i in range(7):
        v = float(deg[i]) if i < len(deg) else 0.0
        lo, hi = limits[i] if i < len(limits) else (-180.0, 180.0)
        out.append(float(max(lo, min(hi, v))))
    return out


def rate_limit(
    target: Sequence[float],
    prev: Sequence[float],
    max_step_deg: float,
) -> List[float]:
    out: List[float] = []
    for i in range(7):
        d = float(target[i]) - float(prev[i])
        if d > max_step_deg:
            d = max_step_deg
        elif d < -max_step_deg:
            d = -max_step_deg
        out.append(float(prev[i]) + d)
    return out


def exp_smooth(
    target: Sequence[float],
    prev: Sequence[float],
    dt_s: float,
    tau_s: float,
) -> List[float]:
    """一阶低通：向目标指数逼近，tau 越大越平滑。"""
    if tau_s <= 1e-6 or dt_s <= 0.0:
        return [float(target[i]) if i < len(target) else 0.0 for i in range(7)]
    alpha = 1.0 - math.exp(-float(dt_s) / float(tau_s))
    alpha = max(0.0, min(1.0, alpha))
    out: List[float] = []
    for i in range(7):
        t = float(target[i]) if i < len(target) else 0.0
        p = float(prev[i]) if i < len(prev) else 0.0
        out.append(p + alpha * (t - p))
    return out


def trajectory_smooth(
    target: Sequence[float],
    prev_pos: Sequence[float],
    prev_vel: Sequence[float],
    dt_s: float,
    max_vel_deg_s: float,
    max_acc_deg_s2: float,
) -> Tuple[List[float], List[float]]:
    """
    发送端轨迹平滑：速度/加速度受限地跟踪目标关节角。
    返回 (new_pos, new_vel)，单位 deg / deg/s。
    """
    dt = max(1e-4, float(dt_s))
    vmax = max(0.0, float(max_vel_deg_s))
    amax = max(0.0, float(max_acc_deg_s2))
    pos: List[float] = []
    vel: List[float] = []
    for i in range(7):
        t = float(target[i]) if i < len(target) else 0.0
        p = float(prev_pos[i]) if i < len(prev_pos) else 0.0
        v = float(prev_vel[i]) if i < len(prev_vel) else 0.0
        err = t - p

        # 刹车距离所需速度；逼近目标时提前减速
        if amax > 1e-9:
            brake_v = math.sqrt(max(0.0, 2.0 * amax * abs(err)))
            desired_v = math.copysign(min(vmax, brake_v), err) if abs(err) > 1e-9 else 0.0
            dv = desired_v - v
            max_dv = amax * dt
            if dv > max_dv:
                dv = max_dv
            elif dv < -max_dv:
                dv = -max_dv
            v = v + dv
        else:
            v = math.copysign(min(vmax, abs(err) / dt), err) if abs(err) > 1e-9 else 0.0

        if vmax > 0.0:
            v = max(-vmax, min(vmax, v))

        step = v * dt
        # 不越过目标（同向跟踪时）
        if (err > 0.0 and step > err) or (err < 0.0 and step < err):
            p = t
            v = 0.0
        else:
            p = p + step
        pos.append(p)
        vel.append(v)
    return pos, vel
