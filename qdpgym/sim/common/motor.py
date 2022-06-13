import collections
from typing import Iterable

import numpy as np
import torch

from qdpgym.sim.common.identify import ActuatorNet
from qdpgym.utils import get_padded, replace_is


class MotorSim(object):
    def __init__(self, frequency):
        self._freq = frequency
        self._observe_done = False
        self._pos = self._vel = 0.
        self._residue = None

        self._joint_lower = self._joint_upper = None
        self._torque_limits = None
        self._cmd_clip = None
        self._input_delay = self._output_delay = 0

        self._torque_history = collections.deque(maxlen=11)
        self._residue_history = collections.deque(maxlen=11)
        self._obs_history = collections.deque(maxlen=11)

    def set_latency(self, input_delay: int = 0, output_delay: int = 0):
        self._input_delay = input_delay
        self._output_delay = output_delay

    def set_joint_limits(self, lower: Iterable = None, upper: Iterable = None):
        lower, upper = list(lower), list(upper)
        if lower is not None and None in lower:
            replace_is(lower, None, -np.inf)
        if upper is not None and None in upper:
            replace_is(upper, None, np.inf)
        self._joint_lower = np.array(lower)
        self._joint_upper = np.array(upper)

    def set_torque_limits(self, upper):
        self._torque_limits = np.asarray(upper)

    def set_cmd_clip(self, upper):
        self._cmd_clip = np.asarray(upper)

    def reset(self):
        self._observe_done = False
        self._torque_history.clear()
        self._residue_history.clear()

    def update_observation(self, pos, vel):
        self._observe_done = True
        self._obs_history.append((np.array(pos), np.array(vel)))
        self._pos, self._vel = get_padded(self._obs_history, -self._input_delay - 1)

    def apply_hybrid(self, des_pos, ff_torque):
        if self._joint_lower is not None or self._joint_upper is not None:
            des_pos = np.clip(des_pos, self._joint_lower, self._joint_upper)
        self._residue = des_pos - self._pos
        if self._cmd_clip:
            self._residue = np.clip(self._residue, -self._cmd_clip, self._cmd_clip)
        self._residue_history.append(self._residue)
        return self.apply_torque(self.calc_torque() + ff_torque)

    def apply_position(self, des_pos):
        return self.apply_hybrid(des_pos, 0)

    def apply_torque(self, des_torque):
        assert self._observe_done, 'Update observation before executing a command'
        if self._torque_limits is not None:
            des_torque = np.clip(des_torque, -self._torque_limits, self._torque_limits)
        self._torque_history.append(des_torque)
        self._observe_done = False
        return get_padded(self._torque_history, -self._output_delay - 1)

    def calc_torque(self):
        raise NotImplementedError


class PdMotorSim(MotorSim):
    def __init__(self, frequency, kp, kd):
        super().__init__(frequency)
        self._kp, self._kd = np.asarray(kp), np.asarray(kd)
        self._kp_part, self._kd_part = 0., 0.

    def calc_torque(self):
        self._kp_part, self._kd_part = self._kp * self._residue, self._kd * self._vel
        return self._kp_part - self._kd_part


class ActuatorNetSim(MotorSim):
    def __init__(self, frequency, net=None):
        super().__init__(frequency)
        self._net = net

    def load_params(self, model_path, device='cpu'):
        model_info = torch.load(model_path, map_location=device)
        self._net = ActuatorNet(hidden_dims=model_info['hidden_dims']).to(device)
        self._net.load_state_dict(model_info['model'])

    def calc_torque(self):
        return self._net.calc_torque(
            self._residue,
            get_padded(self._residue_history, -4),
            get_padded(self._residue_history, -7),
            self._vel,
            get_padded(self._obs_history, -4)[1],
            get_padded(self._obs_history, -7)[1]
        )
