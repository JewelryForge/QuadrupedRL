from __future__ import annotations

import collections
import enum
from typing import Iterable

import numpy as np

from burl.rl.state import MotorState


class MotorMode(enum.Enum):
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    TORQUE = enum.auto()
    # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
    # position and velocities. kp and kd are PD gains. tau is the additional
    # motor torque. This is the most flexible control mode.
    HYBRID = enum.auto()
    PWM = enum.auto()


class MotorSim(object):
    def __init__(self, robot, num=1, **kwargs):
        self._num = num
        self._robot = robot
        self._kp: np.ndarray = np.asarray(kwargs.get('kp', 60))
        self._kd: np.ndarray = np.asarray(kwargs.get('kd', 0.5))
        # assert self._kd.shape == self._kp.shape
        pos_limits: np.ndarray | Iterable | float | None = kwargs.get('pos_limits', None)
        torque_limits: np.ndarray | Iterable | float | None = kwargs.get('torque_limits', 33.5)
        self._frequency = kwargs.get('frequency', 240)
        assert self._frequency > 0
        self._pos, self._vel, self._acc = 0, 0, 0

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
        assert self._mode == MotorMode.POSITION
        self._position_history = collections.deque(maxlen=50)
        self._observe_done = False
        self._kp_part, self._kd_part = 0., 0.

    @property
    def mode(self):
        return self._mode

    def reset(self):
        self._position_history.clear()
        self._observe_done = False

    def update_observation(self, pos=None, vel=None):
        self._observe_done = True
        if pos is not None:
            self._pos = pos
            if vel is not None:
                self._vel = vel
                return
        else:
            try:
                self._pos = np.asarray(self._robot.getJointPositions(noisy=True))
            except AttributeError:
                self._pos = np.asarray(self._robot.joint_states.position)
        self._position_history.append(self._pos)
        ph = self._position_history
        self._vel = (ph[-1] - ph[-2]) * self._frequency if len(ph) > 1 else 0.0
        # self._acc = (ph[-1] + ph[-3] - 2 * ph[-2]) * self._frequency ** 2 if len(ph) > 2 else 0.0
        # return MotorState(position=self._pos, velocity=self._vel, acceleration=self._acc)

    def apply_command(self, command, *args):
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
        self._kp_part, self._kd_part = (des_pos - self._pos) * self._kp, self._kd * self._vel
        return self._set_torque(self._kp * (des_pos - self._pos) - self._kd * self._vel)

    def _set_torque(self, des_torque):
        if hasattr(self, '_torque_limits_upper'):
            return np.clip(des_torque, self._torque_limits_lower, self._torque_limits_upper)
        return des_torque


if __name__ == '__main__':
    pass
