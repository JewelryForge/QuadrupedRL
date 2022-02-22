from __future__ import annotations

import collections
import math

import numpy as np


class MotorSim(object):
    POSITION = 0
    # VELOCITY = 1
    TORQUE = 2
    HYBRID = 3

    def __init__(self, kp, kd, latency_steps=0, pos_limits=None, torque_limits=None):
        self._kp, self._kd = np.asarray(kp), np.asarray(kd)
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

        self._observe_done = False
        self._pos, self._vel = 0., 0.
        self._latency_steps = latency_steps
        self._kp_part, self._kd_part = 0., 0.
        self._torque_history = collections.deque(maxlen=50)

    def reset(self):
        self._observe_done = False

    def update_observation(self, pos, vel):
        self._observe_done = True
        self._pos = np.asarray(pos)
        self._vel = np.asarray(vel)

    def apply_hybrid(self, des_pos, ff_torque):
        if hasattr(self, '_pos_limits_upper'):
            des_pos = np.clip(des_pos, self._pos_limits_lower, self._pos_limits_upper)
        self._kp_part, self._kd_part = self._kp * (des_pos - self._pos), self._kd * self._vel
        return self.apply_torque(self._kp_part - self._kd_part + ff_torque)

    def apply_position(self, des_pos):
        return self.apply_hybrid(des_pos, 0)

    def apply_torque(self, des_torque):
        assert self._observe_done, 'Update observation before executing a command'
        if hasattr(self, '_torque_limits_upper'):
            des_torque = np.clip(des_torque, self._torque_limits_lower, self._torque_limits_upper)
        self._torque_history.append(des_torque)
        self._observe_done = False
        try:
            return self._torque_history[-self._latency_steps - 1]
        except IndexError:
            return self._torque_history[0]


if __name__ == '__main__':
    pass
