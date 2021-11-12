from __future__ import annotations

import collections
import enum
from typing import Iterable

import numpy as np

np.set_printoptions(precision=3, linewidth=10000)


class MotorMode(enum.Enum):
    POSITION = enum.auto()
    VELOCITY = enum.auto()
    TORQUE = enum.auto()
    # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
    # position and velocities. kp and kd are PD gains. tau is the additional
    # motor torque. This is the most flexible control mode.
    HYBRID = enum.auto()
    PWM = enum.auto()


class MotorBase(object):
    def __init__(self, **kwargs):
        self._kp: np.ndarray = np.asarray(kwargs.get('kp', 60))
        self._kd: np.ndarray = np.asarray(kwargs.get('kd', 1))
        pos_limits: np.ndarray | Iterable | float | None = kwargs.get('pos_limits', None)
        torque_limits: np.ndarray | Iterable | float | None = kwargs.get('torque_limits', 33.5)
        assert self._kd.shape == self._kp.shape
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

    def update_observation(self, time, observation):
        self._observe_done = True
        observation = np.asarray(observation)
        # assert observation.shape == self._shape
        self._observation_history.append((time, observation))

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
        t, pos = self._observation_history[-1]
        try:
            t_p, pos_p = self._observation_history[-2]
            vel = (pos - pos_p) / (t - t_p)
        except IndexError:
            vel = 0
        return self._set_torque(self._kp * (des_pos - pos) - self._kd * vel)

    def _set_torque(self, des_torque):
        if hasattr(self, '_torque_limits_upper'):
            return np.clip(des_torque, self._torque_limits_lower, self._torque_limits_upper)
        return des_torque

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if isinstance(m, MotorMode):
            self._mode = m
        else:
            print(f'Cannot set {m} as motor mode')


if __name__ == '__main__':
    print(isinstance(2, MotorMode))
