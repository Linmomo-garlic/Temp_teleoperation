"""
Multi-Port Dynamixel Driver

Supports Dynamixel servos distributed across multiple COM/serial ports.
Each port gets its own DynamixelDriver instance with its own reading thread.
The MultiPortDynamixelDriver presents a unified interface matching
DynamixelDriver's protocol, merging data from all ports transparently.

Usage:
    port_config = {
        "COM9": [1, 2, 3, 4, 5, 6, 7],
    }
    servo_types_map = {
        1: "XM430_W210_T", 2: "XM430_W210_T",
        3: "XC330_T288_T", 4: "XC330_T288_T",
        5: "XC330_T288_T", 6: "XC330_T288_T",
        7: "XC330_T288_T",
    }
    driver = MultiPortDynamixelDriver(
        port_config=port_config,
        global_ids=[1, 2, 3, 4, 5, 6, 7],
        servo_types_map=servo_types_map,
    )
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gello_leader.dynamixel.driver import DynamixelDriver


class MultiPortDynamixelDriver:
    """Manages multiple Dynamixel drivers across different COM/serial ports.

    Each port has its own DynamixelDriver instance with its own reading thread.
    This class routes reads (get_positions_and_velocities) and writes
    (set_torque, set_torque_mode, set_operating_mode) to the appropriate
    port driver, presenting a unified interface ordered by global_ids.
    """

    def __init__(
        self,
        port_config: Dict[str, List[int]],
        global_ids: Sequence[int],
        servo_types_map: Dict[int, str],
        baudrate: int = 1000000,
    ):
        """Initialize multi-port Dynamixel driver.

        Args:
            port_config: Mapping from port name (e.g. "COM9") to list of
                         Dynamixel IDs on that port.
            global_ids: Ordered list of all servo IDs across all ports.
                        This determines the output order of
                        get_positions_and_velocities() and the expected
                        input order of set_torque().
            servo_types_map: Mapping from servo ID to servo model string
                             (e.g. "XC330_T288_T", "XM430_W210_T").
            baudrate: Baudrate for all ports (default 1000000).
        """
        self._port_config = dict(port_config)
        self._global_ids = list(global_ids)
        self._servo_types_map = dict(servo_types_map)
        self._baudrate = baudrate

        # Build lookup tables: global ID -> (port_name, local_index)
        self._id_to_port: Dict[int, str] = {}
        self._id_to_local_idx: Dict[int, int] = {}
        for port_name, ids in port_config.items():
            for local_idx, servo_id in enumerate(ids):
                self._id_to_port[servo_id] = port_name
                self._id_to_local_idx[servo_id] = local_idx

        # Validate that all global_ids are covered by port_config
        for gid in self._global_ids:
            if gid not in self._id_to_port:
                raise ValueError(
                    f"ID {gid} in global_ids not found in any port_config port"
                )

        # Create one DynamixelDriver per port
        self._drivers: Dict[str, DynamixelDriver] = {}
        self._num_motors = len(self._global_ids)

        print(f"MultiPortDynamixelDriver: connecting across {len(port_config)} ports")
        for port_name, ids in port_config.items():
            port_servo_types = [servo_types_map[sid] for sid in ids]
            print(f"  Port {port_name}: IDs={ids}, types={port_servo_types}")
            try:
                # Windows FTDI: avoid \\.\ prefix issues on COM10+, and give
                # a short settle time between opening multiple U2D2 adapters.
                open_port = port_name
                if port_name.upper().startswith("COM"):
                    try:
                        com_num = int(port_name.upper().replace("COM", ""))
                        if com_num >= 10:
                            open_port = f"\\\\.\\{port_name}"
                    except ValueError:
                        open_port = port_name

                self._drivers[port_name] = DynamixelDriver(
                    ids=ids,
                    servo_types=port_servo_types,
                    port=open_port,
                    baudrate=baudrate,
                    use_fake_fallback=False,
                )
                print(f"  [OK] Connected to {port_name}")
                time.sleep(0.2)
            except Exception as e:
                # Clean up already-opened ports on failure
                for p, d in self._drivers.items():
                    try:
                        d.close()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Failed to connect to port {port_name}: {e}"
                ) from e

    # ---- Read methods ----

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions (rad) and velocities (rad/s) for all servos.

        Data is collected from all port drivers and merged into a single
        array ordered by global_ids.

        Returns:
            Tuple of (positions, velocities), each a 1D numpy array.
        """
        # Collect from all ports
        port_positions: Dict[str, np.ndarray] = {}
        port_velocities: Dict[str, np.ndarray] = {}

        for port_name, driver in self._drivers.items():
            pos, vel = driver.get_positions_and_velocities()
            port_positions[port_name] = pos
            port_velocities[port_name] = vel

        # Merge into global order
        global_pos = np.zeros(self._num_motors, dtype=np.float64)
        global_vel = np.zeros(self._num_motors, dtype=np.float64)

        for global_idx, servo_id in enumerate(self._global_ids):
            port_name = self._id_to_port[servo_id]
            local_idx = self._id_to_local_idx[servo_id]
            global_pos[global_idx] = port_positions[port_name][local_idx]
            global_vel[global_idx] = port_velocities[port_name][local_idx]

        return global_pos, global_vel

    def get_joints(self) -> np.ndarray:
        """Get current joint angles (rad) for all servos, in global_ids order."""
        pos, _ = self.get_positions_and_velocities()
        return pos

    # ---- Write methods ----

    def set_torque(self, torques: Sequence[float]) -> None:
        """Set joint torques (Nm) for all servos, ordered by global_ids.

        Torques are split per-port and dispatched to the appropriate
        sub-driver. Each sub-driver handles Nm->mA conversion based on
        its own servo type mappings.

        Args:
            torques: Torque values in Nm, one per global ID, in
                     global_ids order.
        """
        if len(torques) != self._num_motors:
            raise ValueError(
                f"Expected {self._num_motors} torque values, got {len(torques)}"
            )

        for port_name, driver in self._drivers.items():
            port_ids = self._port_config[port_name]
            port_torques = []
            for servo_id in port_ids:
                global_idx = self._global_ids.index(servo_id)
                port_torques.append(float(torques[global_idx]))
            driver.set_torque(port_torques)

    def set_current(self, currents: Sequence[float]) -> None:
        """Set motor currents (mA) for all servos, ordered by global_ids."""
        if len(currents) != self._num_motors:
            raise ValueError(
                f"Expected {self._num_motors} current values, got {len(currents)}"
            )

        for port_name, driver in self._drivers.items():
            port_ids = self._port_config[port_name]
            port_currents = []
            for servo_id in port_ids:
                global_idx = self._global_ids.index(servo_id)
                port_currents.append(float(currents[global_idx]))
            driver.set_current(port_currents)

    def set_joints(self, joint_angles: Sequence[float]) -> None:
        """Set joint angles (rad) for all servos, ordered by global_ids."""
        if len(joint_angles) != self._num_motors:
            raise ValueError(
                f"Expected {self._num_motors} joint angles, got {len(joint_angles)}"
            )

        for port_name, driver in self._drivers.items():
            port_ids = self._port_config[port_name]
            port_angles = []
            for servo_id in port_ids:
                global_idx = self._global_ids.index(servo_id)
                port_angles.append(float(joint_angles[global_idx]))
            driver.set_joints(port_angles)

    # ---- Configuration methods (broadcast to all ports) ----

    def set_torque_mode(self, enable: bool) -> None:
        """Enable or disable torque on all servos across all ports."""
        for port_name, driver in self._drivers.items():
            driver.set_torque_mode(enable)

    def set_operating_mode(self, mode: int) -> None:
        """Set operating mode on all servos across all ports.

        Args:
            mode: 0 for current control, 3 for position control.
        """
        for port_name, driver in self._drivers.items():
            driver.set_operating_mode(mode)

    def torque_enabled(self) -> bool:
        """Check if torque is enabled (checks first driver only)."""
        if self._drivers:
            first_driver = next(iter(self._drivers.values()))
            return first_driver.torque_enabled()
        return False

    # ---- Lifecycle ----

    def close(self) -> None:
        """Close all port drivers and stop their reading threads."""
        for port_name, driver in self._drivers.items():
            try:
                driver.close()
            except Exception as e:
                print(f"Warning: error closing driver on {port_name}: {e}")
        self._drivers.clear()
        print("MultiPortDynamixelDriver: all ports closed")

    @property
    def num_motors(self) -> int:
        return self._num_motors

    @property
    def global_ids(self) -> List[int]:
        return list(self._global_ids)
