import collections
from typing import Deque, Optional

import numpy as np

from qdpgym import tf, utils as ut
from qdpgym.sim.abc import Quadruped, QuadrupedHandle, Snapshot


class NoisyHandle(QuadrupedHandle):
    def __init__(self, robot: Quadruped, frequency, latency=0):
        self._robot = robot
        self._freq = frequency
        self._latency = latency

        self._obs: Optional[Snapshot] = None
        self._obs_buffer: Deque[Snapshot] = collections.deque(maxlen=20)
        self._obs_history: Deque[Snapshot] = collections.deque(maxlen=100)

    @property
    def raw(self):
        return self._robot

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    @property
    def obs_history(self):
        return ut.PadWrapper(self._obs_history)

    @property
    def cmd_history(self):
        return self._robot.cmd_history

    @property
    def not_delayed(self):
        if self._obs_buffer:
            return self._obs_buffer[-1]
        return self._obs

    def update_observation(self, state: Snapshot, random_state):
        """Add noise on observation"""
        add_noise = random_state.normal
        obs = Snapshot(
            rpy=add_noise(state.rpy, 1e-2),
            velocimeter=add_noise(state.velocimeter, 5e-2),
            gyro=add_noise(state.gyro, 5e-2),
            joint_pos=add_noise(state.joint_pos, 5e-3),
            joint_vel=add_noise(state.joint_vel, 1e-1)
        )
        obs.orientation = tf.Quaternion.from_rpy(obs.rpy)
        self._obs_buffer.append(obs)
        if len(self._obs_buffer) <= self._latency * self._freq:
            self._obs = self._obs_buffer[0]
        else:
            self._obs = self._obs_buffer.popleft()
        self._obs_history.append(self._obs)

    def get_base_pos(self) -> np.ndarray:
        return self._obs.position

    def get_base_orn(self) -> np.ndarray:
        return self._obs.orientation

    def get_base_rot(self) -> np.ndarray:
        return tf.Rotation.from_quaternion(self._obs.orientation)

    def get_base_rpy(self) -> np.ndarray:
        return self._obs.rpy

    def get_base_rpy_rate(self) -> np.ndarray:
        return tf.get_rpy_rate_from_ang_vel(self._obs.rpy, self._obs.angular_vel)

    def get_base_lin(self) -> np.ndarray:
        return self._obs.linear_vel

    def get_base_ang(self) -> np.ndarray:
        return self._obs.angular_vel

    def get_joint_pos(self) -> np.ndarray:
        return self._obs.joint_pos

    def get_joint_vel(self) -> np.ndarray:
        return self._obs.joint_vel

    def get_velocimeter(self):
        return self._obs.velocimeter

    def get_gyro(self):
        return self._obs.gyro

    def get_accelerometer(self):
        return self._obs.accelerometer

    def get_torso_contact(self):
        return self._robot.get_torso_contact()

    def get_leg_contacts(self):
        return self._robot.get_leg_contacts()

    def get_last_torque(self) -> np.ndarray:
        return self._robot.get_last_torque()

    def get_state_history(self, latency):
        return ut.get_padded(self._obs_history, -int(latency * self._freq) - 1)

    def get_cmd_history(self, latency):
        return self._robot.get_cmd_history(latency)

    def get_foot_contacts(self):
        return self._robot.get_foot_contacts()

    def get_contact_forces(self):
        return self._robot.get_contact_forces()

    def get_force_sensor(self):
        return self._robot.get_force_sensor()

    def get_last_command(self) -> np.ndarray:
        return self._robot.get_last_command()

    def reset(self):
        self._obs = None
        self._obs_buffer.clear()
        self._obs_history.clear()
