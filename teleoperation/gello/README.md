# Gello ↔ 天机（Tianji）遥操作

Windows 本机读取 / 驱动 Gello Dynamixel 主臂，经 **Thor / Jetson** 上的 `thor_joint_agent.py` TCP 代理与天机双臂交换关节角，并支持定点关节控制、关节阻抗使能、主臂力反馈（含从臂 `joint_ext` 映射）。

> 目标路径（本仓库）：`teleoperation/gello/`

---

## 1. 系统架构

```text
┌──────────────────────────┐         TCP JSON :15666        ┌─────────────────────────┐
│  Windows PC (GUI)        │ ─────────────────────────────▶ │  Thor / Jetson          │
│  - Dynamixel COM9        │ ◀───────────────────────────── │  thor_joint_agent.py    │
│  - 标定 / 跟随 / 力反馈   │                                 │  eth1 → 天机柜 SDK      │
└──────────────────────────┘                                 └───────────┬─────────────┘
                                                                         │
                                                                         ▼
                                                               Tianji Cabinet
                                                               192.168.1.190
```

- **本机不直连天机柜**。所有 `connect / enable / set_joints / get_state` 均经 Thor 代理转发。
- 主臂串口默认：`COM9`，舵机 ID `1..7` ↔ 天机 `J1..J7`。

---

## 2. 功能一览

| 功能 | 说明 |
|------|------|
| 主控从 | Gello → 天机：读主臂角，`set_joints` 控从臂；主臂松力矩 |
| 从控主 | 天机 → Gello：`get_state` 读从臂角，写主臂位置模式 |
| 定点关节角 | GUI 输入 J1..J7，经代理插值下发 |
| 从臂模式 | `position`（位置跟随）/ `impedance`（关节阻抗） |
| 统一频率 | GUI 设置 `frequency_hz`，同步上位机跟随环与代理插值环 |
| 力反馈 | 手动 `J×增益`，或从臂 `joint_ext × 增益` 映射到主臂电流力矩模式 |
| 主臂锁力 | 扩展位置模式开力矩，锁住当前姿（抗重力）；「松开主臂」关力矩可手掰 |
| 日志 | 每次启动覆盖写入 `log/teleop_gui.log` |

---

## 3. 目录结构

```text
teleoperation/gello/
  README.md                     # 本指南
  gello_tianji_teleop_gui.py    # Windows GUI 入口
  run_teleop.bat                # 一键启动（Windows）
  configs/teleop.yaml           # 标定与控制参数（勿提交真实密码）
  gello_leader/
    agent_client.py             # Thor TCP 客户端 + SSH 拉起代理
    mapping.py                  # 标定 / 限速 / 平滑
    dynamixel/                  # Dynamixel 驱动（位置 / 电流力矩）
  thor/
    thor_joint_agent.py         # 部署到 Jetson 的代理脚本
```

---

## 4. 环境依赖

### Windows（GUI）

```bash
conda activate new_gello   # 或你的环境名
pip install numpy pyyaml pyserial paramiko
# dynamixel_sdk：建议安装官方 SDK 的 python 包，或 editable 安装 DynamixelSDK/python
```

启动：

```bash
cd teleoperation/gello
python gello_tianji_teleop_gui.py
# 或双击 run_teleop.bat（需改 bat 内 python 路径）
```

### Thor / Jetson（代理）

1. 将 `thor/thor_joint_agent.py` 同步到 Jetson 工程，例如：  
   `~/Desktop/lambda2_jetson_control/jetson_control/scripts/thor_joint_agent.py`
2. 保证该环境可 `import` 天机 SDK / `tjfx_common`（与现有 `jetson_control` 一致）。
3. 监听：

```bash
cd ~/Desktop/lambda2_jetson_control/jetson_control
python3 scripts/thor_joint_agent.py --port 15666
```

GUI「启动/探测 Thor 代理」也可经 SSH 自动同步并后台拉起（需在 `teleop.yaml` 填写 `thor.ssh_user/ssh_pass`）。

---

## 5. 网络与配置

编辑 `configs/teleop.yaml`：

| 字段 | 含义 | 示例 |
|------|------|------|
| `thor.host` | Thor IP | `172.20.10.4` |
| `thor.agent_port` | 代理端口 | `15666` |
| `tianji.robot_ip` | 天机柜 IP（仅 Thor 可达） | `192.168.1.190` |
| `tianji.control_mode` | `position` / `impedance` | `impedance` |
| `control.frequency_hz` | 控制频率（本机+代理） | `100` |
| `dynamixel.joint_signs` | 轴方向 | J2/J6 常为 `-1` |
| `control.force_gains` | 手动/从臂外力→主臂增益 | 见下节 |

**安全**：上传/分享前请把 `ssh_pass` 换成占位符；勿提交真实 token / 密码。

---

## 6. 推荐操作流程

### 6.1 公共准备

1. PC 能 `ping` Thor；Thor `eth1` 能 `ping` 天机柜。
2. 打开 GUI → **连接主臂**。
3. **启动/探测 Thor 代理** → **连接代理** → **连接天机柜**。
4. 主臂摆到与天机一致的 Home → **当前姿设为零位**。
5. 若某轴反向：改 `joint_signs` 或界面 sign → **应用全部 sign**。

### 6.2 主控从（遥操作）

1. 方向选 **主控从**，先勾 **干跑**，确认标定目标角合理。
2. 从臂模式选 **关节阻抗**（或位置跟随）→ **使能**。
3. **同步天机反馈→限速基准** → 取消干跑 → **开始跟随**。
4. 控制频率在 GUI 修改后点 **应用并同步到Jetson**。

### 6.3 定点关节角

1. 使能后，在「定点关节角」填 J1..J7 或「读天机反馈填入」。
2. 设置插值时间（大角度建议 ≥2 s）→ **下发到天机**。

### 6.4 力反馈 → 主臂

主臂进入 **电流力矩模式**，单位 Nm。

| 力源 | 行为 |
|------|------|
| 手动输入 | `τ ≈ ±gain × J1..J7`（与从臂共用 G1..G7） |
| 从臂外力(joint_ext) | `τ ≈ ±gain × joint_ext`（需从臂关节阻抗） |

- **反馈时间 s**：到点自动停；`0` = 直到手动停止。
- **透明/提示**：`|joint_ext|`（或手动非零力矩）低于死区时主臂**关力矩**保持丝滑；超过死区才开电流回力。退出阈值≈死区×`force_exit_deadband_ratio`（默认 0.6）。
- 可与 **主控从** 并行（从臂外力模式）；与 **从控主** 互斥。
- 限幅建议 ≤ `0.8 Nm`（XC330 接近上限；J1/J2 的 XM430 可略高）。

**推荐增益起点**（再按手感微调）：

```text
G1..G7 ≈ 0.10, 0.08, 0.10, 0.12, 0.15, 0.15, 0.18
```

经验：先 0.05–0.1；空载抖动则加大死区（0.3–1.0 Nm）。

---

## 7. 标定公式

正向（主控从）：

```text
q_cmd = (q_raw - joint_offsets_rad) * joint_signs   # rad
q_cmd_deg = rad2deg(q_cmd) 再按软限位裁剪
```

反向（从控主）：

```text
q_raw = deg2rad(q_tianji) / joint_signs + joint_offsets_rad
```

力映射：

```text
τ_master = τ_cmd * joint_signs
# 手动输入：τ_cmd = (±) force_gains[i] * J[i]          （再缩放/限幅）
# 从臂外力：τ_cmd = (±) force_gains[i] * joint_ext[i]  （再缩放/限幅）
```

---

## 8. 代理协议（摘要）

一行一个 JSON 请求/响应（UTF-8，`\n` 结尾），常用命令：

- `ping` / `connect` / `disconnect` / `get_state`
- `set_params`：`frequency_hz`, `control_mode`, `max_step_deg`, `joint_k/d` …
- `enable`：`arm`, `vel`, `acc`, `mode=position|impedance`, `K`, `D`
- `set_joints`：`arm`, `joints[7]`（deg）, `interp_s`
- `sync_target` / `soft_stop` / `disable`

---

## 9. 常见问题

1. **`No module named dynamixel_sdk`**  
   在 `new_gello` 环境正确安装 Dynamixel SDK Python 包；editable 安装路径勿失效。

2. **连接柜 / 使能 timed out**  
   本机到 Thor 延迟较高时，GUI 已加长超时；确认 Thor→`192.168.1.190` 通，代理未卡在旧 enable。

3. **频率上不去**  
   单次 `set_joints`/`get_state` RTT 若 >10 ms，有效频率受网络限制；可降到 50–100 Hz，并减少热路径上的 `get_state`。

4. **J2/J6 方向反了**  
   `joint_signs` 对应位置设为 `-1`。

---

## 10. 安全注意

- 实机前先 **干跑**；周围留空；急停可用。
- 力反馈从小力矩（0.05–0.2 Nm）试起，勿长时间顶满堵转电流。
- 关节阻抗下人机接近时注意碰撞与外力突变。

---

## 11. 许可证 / 归属

Lambda Robotics 内部遥操作模块。部署时请遵循公司 SDK 与硬件安全规范。
