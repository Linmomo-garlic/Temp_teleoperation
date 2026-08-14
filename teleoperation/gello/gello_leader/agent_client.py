"""TCP client for Thor thor_joint_agent.py."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

DEFAULT_THOR_HOST = "172.20.10.4"
DEFAULT_AGENT_PORT = 15666
DEFAULT_SSH_USER = "lambda2"
DEFAULT_SSH_PASS = "lambda"
AGENT_REMOTE = (
    "/home/lambda2/Desktop/lambda2_jetson_control/jetson_control/"
    "scripts/thor_joint_agent.py"
)
AGENT_CWD = "/home/lambda2/Desktop/lambda2_jetson_control/jetson_control"
SESSION_REMOTE = (
    "/home/lambda2/Desktop/lambda2_jetson_control/jetson_control/"
    "tjfx_common/robot_session.py"
)


class AgentClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._buf = bytearray()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        self.close()
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        self._sock = s
        self._buf.clear()

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._buf.clear()

    def call(self, cmd: str, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        req = {"cmd": cmd, **kwargs}
        line = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if not self._sock:
                raise RuntimeError("未连接代理")
            old_timeout = self._sock.gettimeout()
            if timeout is not None:
                self._sock.settimeout(timeout)
            try:
                self._sock.sendall(line)
                while True:
                    idx = self._buf.find(b"\n")
                    if idx >= 0:
                        raw = bytes(self._buf[:idx]).decode("utf-8", errors="replace")
                        del self._buf[: idx + 1]
                        return json.loads(raw)
                    chunk = self._sock.recv(8192)
                    if not chunk:
                        self.close()
                        raise RuntimeError("代理连接断开")
                    self._buf.extend(chunk)
            finally:
                if self._sock is not None:
                    self._sock.settimeout(old_timeout)


def ensure_agent_running(
    host: str,
    port: int,
    user: str,
    password: str,
    log: Callable[[str], None],
    local_agent_path: Optional[str] = None,
    force_restart: bool = False,
) -> bool:
    """若代理未监听，则经 SSH 在 Thor 后台启动，并可同步本仓库 thor 脚本。"""
    if not force_restart:
        try:
            c = AgentClient(host, port, timeout=2.0)
            c.connect()
            r = c.call("ping")
            c.close()
            if r.get("ok"):
                log(f"代理已在线 {host}:{port}")
                return True
        except Exception:
            pass

    log(f"代理未监听，SSH 启动 {user}@{host} ..." if not force_restart else f"强制重启代理 {user}@{host} ...")
    try:
        import paramiko
    except ImportError:
        log("缺少 paramiko，请先手动在 Thor 启动代理")
        return False

    here = os.path.dirname(os.path.abspath(__file__))
    gello_root = os.path.dirname(here)
    repo_root = os.path.dirname(os.path.dirname(gello_root))
    if local_agent_path is None:
        local_agent_path = os.path.join(gello_root, "thor", "thor_joint_agent.py")
    local_session_path = os.path.join(
        repo_root, "jetson_control", "tjfx_common", "robot_session.py"
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password, timeout=15)
        sftp = client.open_sftp()
        try:
            sftp.put(local_agent_path, AGENT_REMOTE)
            log(f"已同步代理脚本 → {AGENT_REMOTE}")
            if os.path.isfile(local_session_path):
                sftp.put(local_session_path, SESSION_REMOTE)
                log(f"已同步会话封装 → {SESSION_REMOTE}")
        except Exception as e:
            log(f"同步脚本失败(将尝试用远端已有文件): {e}")
        finally:
            sftp.close()

        kill_cmd = "pkill -f thor_joint_agent.py || true"
        _, ko, ke = client.exec_command(kill_cmd, timeout=8)
        ko.read()
        ke.read()
        time.sleep(0.4)

        start_cmd = (
            "python3 - <<'PY'\n"
            "import subprocess, sys\n"
            "logf = open('/tmp/thor_joint_agent.log', 'w')\n"
            "p = subprocess.Popen(\n"
            f"    [sys.executable, {AGENT_REMOTE!r}, '--port', str({port})],\n"
            f"    cwd={AGENT_CWD!r},\n"
            "    stdout=logf, stderr=subprocess.STDOUT,\n"
            "    stdin=subprocess.DEVNULL, start_new_session=True)\n"
            "print(p.pid)\n"
            "PY"
        )
        _, stdout, stderr = client.exec_command(start_cmd, timeout=15)
        pid = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        log(f"代理已后台启动 pid={pid or '?'} {err}")
    except Exception as e:
        log(f"SSH 启动失败: {e}")
        return False
    finally:
        client.close()

    for _ in range(15):
        time.sleep(0.4)
        try:
            c = AgentClient(host, port, timeout=2.0)
            c.connect()
            r = c.call("ping")
            c.close()
            if r.get("ok"):
                log("代理就绪")
                return True
        except Exception:
            continue
    log("代理启动超时，请检查 /tmp/thor_joint_agent.log")
    return False
