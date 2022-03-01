from __future__ import annotations

import collections

import numpy as np
import torch

from burl.sim.ident import ActuatorNet, ActuatorNetWithHistory


class MotorSim(object):
    def __init__(self, frequency=500, input_latency=0., output_latency=0., torque_limits=None, cmd_clip=None):
        if torque_limits:
            torque_limits = np.asarray(torque_limits)
            if not torque_limits.shape:
                assert torque_limits > 0
                self._min_torque, self._max_torque = -torque_limits, torque_limits
            else:
                self._min_torque = np.asarray(torque_limits[0])
                self._max_torque = np.asarray(torque_limits[1])
                assert all(self._min_torque < self._max_torque)

        self._cmd_clip = cmd_clip
        self._observe_done = False
        self._pos, self._vel, self._residue = 0., 0., None
        self._frequency = frequency
        self._input_latency_steps = int(input_latency * frequency)
        self._output_latency_steps = int(output_latency * frequency)
        self._torque_history = collections.deque(maxlen=11)
        self._residue_history = collections.deque(maxlen=11)
        self._observation_history = collections.deque(maxlen=11)

    def reset(self):
        self._observe_done = False
        self._torque_history.clear()
        self._residue_history.clear()

    def update_observation(self, pos, vel):
        self._observe_done = True
        self._observation_history.append((np.array(pos), np.array(vel)))
        if len(self._observation_history) > self._input_latency_steps:
            observation = self._observation_history[-self._input_latency_steps - 1]
        else:
            observation = self._observation_history[0]
        self._pos, self._vel = observation

    def apply_hybrid(self, des_pos, ff_torque):
        self._residue = des_pos - self._pos
        if self._cmd_clip:
            self._residue = np.clip(self._residue, -self._cmd_clip, self._cmd_clip)
        self._residue_history.append(self._residue)
        return self.apply_torque(self.calc_torque() + ff_torque)

    def apply_position(self, des_pos):
        return self.apply_hybrid(des_pos, 0)

    def apply_torque(self, des_torque):
        assert self._observe_done, 'Update observation before executing a command'
        if hasattr(self, '_min_torque'):
            des_torque = np.clip(des_torque, self._min_torque, self._max_torque)
        self._torque_history.append(des_torque)
        self._observe_done = False
        if len(self._torque_history) > self._output_latency_steps:
            return self._torque_history[-self._output_latency_steps - 1]
        else:
            return self._torque_history[0]

    def calc_torque(self):
        raise NotImplementedError


class PdMotorSim(MotorSim):
    def __init__(self, kp, kd, frequency=500, input_latency=0., output_latency=0., torque_limits=None, cmd_clip=None):
        super().__init__(frequency, input_latency, output_latency, torque_limits, cmd_clip)
        self._kp, self._kd = np.asarray(kp), np.asarray(kd)
        self._kp_part, self._kd_part = 0., 0.

    def calc_torque(self):
        self._kp_part, self._kd_part = self._kp * self._residue, self._kd * self._vel
        return self._kp_part - self._kd_part


class ActuatorNetSim(MotorSim):
    def __init__(self, model_path, device='cpu', frequency=500, input_latency=0., output_latency=0.,
                 torque_limits=None, cmd_clip=0.2):
        super().__init__(frequency, input_latency, output_latency, torque_limits, cmd_clip)
        model_info = torch.load(model_path, map_location={'cuda:0': device})
        self.net = ActuatorNet(hidden_dims=model_info['hidden_dims']).to(device)
        self.net.load_state_dict(model_info['model'])

    def calc_torque(self):
        last_residue = self._residue_history[-2] if len(self._residue_history) > 1 else self._residue
        residue_rate = (self._residue - last_residue) * self._frequency
        return self.net.calc_torque(self._residue, residue_rate, self._vel)


class ActuatorNetWithHistorySim(MotorSim):
    def __init__(self, model_path, device='cpu', frequency=500, input_latency=0., output_latency=0.,
                 torque_limits=None, cmd_clip=0.2):
        super().__init__(frequency, input_latency, output_latency, torque_limits, cmd_clip)
        model_info = torch.load(model_path, map_location={'cuda:0': device})
        self.net = ActuatorNetWithHistory(hidden_dims=model_info['hidden_dims']).to(device)
        self.net.load_state_dict(model_info['model'])

    def calc_torque(self):
        return self.net.calc_torque(self._residue,
                                    self.safely_getitem(self._residue_history, -6),
                                    self.safely_getitem(self._residue_history, -11),
                                    self._vel,
                                    self.safely_getitem(self._observation_history, -6)[1],
                                    self.safely_getitem(self._observation_history, -11)[1])

    @staticmethod
    def safely_getitem(seq, idx):
        if idx < 0:
            if len(seq) < -idx:
                return seq[0]
        elif len(seq) <= idx:
            return seq[-1]
        return seq[idx]


if __name__ == '__main__':
    pass
