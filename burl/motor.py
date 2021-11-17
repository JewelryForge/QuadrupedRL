from __future__ import annotations

import collections
import enum
from typing import Iterable

import numpy as np
from burl.bc import Observable
from burl.sensors import MotorEncoder, MotorEncoderDiff, MotorEncoderDiff2
from burl.utils import make_cls


class MotorMode(enum.Enum):
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    TORQUE = enum.auto()
    # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
    # position and velocities. kp and kd are PD gains. tau is the additional
    # motor torque. This is the most flexible control mode.
    HYBRID = enum.auto()
    PWM = enum.auto()


class MotorSim(Observable):
    # Observation dim is automatically given by 'num' attribute
    ALLOWED_SENSORS = {MotorEncoder, MotorEncoderDiff, MotorEncoderDiff2}

    def __init__(self, robot, num=1, **kwargs):
        # self._num = num  # TODO: check if necessary
        self._robot = robot
        self._kp: np.ndarray = np.asarray(kwargs.get('kp', 60))
        self._kd: np.ndarray = np.asarray(kwargs.get('kd', 1))
        assert self._kd.shape == self._kp.shape
        pos_limits: np.ndarray | Iterable | float | None = kwargs.get('pos_limits', None)
        torque_limits: np.ndarray | Iterable | float | None = kwargs.get('torque_limits', 33.5)
        self._frequency = kwargs.get('frequency', 240)
        assert self._frequency > 0
        self._pos, self._vel, self._acc = 0, 0, 0
        _make_sensors = (make_cls(s, dim=num) for s in kwargs.get('make_sensors', ()))
        super().__init__(_make_sensors)

        if pos_limits:
            pos_limits = np.asarray(pos_limits)
            if not pos_limits.shape:
                assert pos_limits > 0
                self._pos_limits_lower, self._pos_limits_upper = -pos_limits, pos_limits
            else:
                self._pos_limits_lower = np.asarray(pos_limits[0])
                self._pos_limits_upper = np.asarray(pos_limits[1])
                assert all(self._pos_limits_lower < self._pos_limits_upper)
        if torque_limits:
            torque_limits = np.asarray(torque_limits)
            if not torque_limits.shape:
                assert torque_limits > 0
                self._torque_limits_lower, self._torque_limits_upper = -torque_limits, torque_limits
            else:
                self._torque_limits_lower = np.asarray(torque_limits[0])
                self._torque_limits_upper = np.asarray(torque_limits[1])
                assert all(self._torque_limits_lower < self._torque_limits_upper)

        self._mode = kwargs.get('mode', MotorMode.POSITION)
        # self._command_history = collections.deque(maxlen=50)
        self._observation_history = collections.deque(maxlen=50)
        self._observe_done = False

    @property
    def mode(self):
        return self._mode

    @property
    def frequency(self):
        return self._frequency

    def reset(self):
        self._observation_history.clear()
        self._observe_done = False

    def _on_update_observation(self):
        self._observe_done = True
        observation = np.array([js.pos for js in self._robot.get_joint_states()])
        self._observation_history.append(observation)
        self._pos = observation
        oh = self._observation_history
        self._vel = (oh[-1] - oh[-2]) * self._frequency if len(oh) > 1 else 0
        self._acc = (oh[-1] + oh[-3] - 2 * oh[-2]) * self._frequency ** 2 if len(oh) > 2 else 0

    def set_command(self, command, *args):
        if not self._observe_done:
            raise RuntimeError('Get a new observation before executing a new command')
        if self._mode == MotorMode.POSITION:
            return self._set_position(command)
        if self._mode == MotorMode.TORQUE:
            return self._set_torque(command)
        raise NotImplementedError

    def _set_position(self, des_pos):
        if hasattr(self, '_pos_limits_upper'):
            des_pos = np.clip(des_pos, self._pos_limits_lower, self._pos_limits_upper)
        return self._set_torque(self._kp * (des_pos - self._pos) - self._kd * self._vel)

    def _set_torque(self, des_torque):
        if hasattr(self, '_torque_limits_upper'):
            return np.clip(des_torque, self._torque_limits_lower, self._torque_limits_upper)
        return des_torque

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_acceleration(self):
        return self._acc


if __name__ == '__main__':
    m = MotorSim()
    print(m.__class__.__name__)
