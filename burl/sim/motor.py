from __future__ import annotations

import collections
import os

import numpy as np
import torch

import burl
from burl.sim.identify import ActuatorNet, ActuatorNetWithHistory
from burl.utils import make_part

__all__ = ['MotorSim', 'PdMotorSim', 'ActuatorNetSim', 'ActuatorNetWithHistorySim', 'ActuatorNetManager']


class MotorSim(object):
    def __init__(self, frequency=500, input_latency=0., output_latency=0.,
                 joint_limits=None, torque_limits=None, cmd_clip=None):
        if joint_limits is not None:
            joint_limits = np.asarray(joint_limits)
            self._joint_lower = np.asarray(joint_limits[:, 0])
            self._joint_upper = np.asarray(joint_limits[:, 1])
            assert all(self._joint_lower < self._joint_upper)

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
        if hasattr(self, '_joint_lower'):
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
    def __init__(self, kp, kd, frequency=500, input_latency=0., output_latency=0.,
                 joint_limits=None, torque_limits=None, cmd_clip=None):
        super().__init__(frequency, input_latency, output_latency, joint_limits, torque_limits, cmd_clip)
        self._kp, self._kd = np.asarray(kp), np.asarray(kd)
        self._kp_part, self._kd_part = 0., 0.

    def calc_torque(self):
        self._kp_part, self._kd_part = self._kp * self._residue, self._kd * self._vel
        return self._kp_part - self._kd_part


class ActuatorNetSim(MotorSim):
    def __init__(self, net, frequency=500, input_latency=0., output_latency=0.,
                 joint_limits=None, torque_limits=None, cmd_clip=0.2):
        super().__init__(frequency, input_latency, output_latency, joint_limits, torque_limits, cmd_clip)
        self.net = net

    def calc_torque(self):
        last_residue = self._residue_history[-2] if len(self._residue_history) > 1 else self._residue
        residue_rate = (self._residue - last_residue) * self._frequency
        return self.net.calc_torque(self._residue, residue_rate, self._vel)


class ActuatorNetWithHistorySim(ActuatorNetSim):
    def __init__(self, net, frequency=500, input_latency=0., output_latency=0.,
                 joint_limits=None, torque_limits=None, cmd_clip=0.2):
        super().__init__(net, frequency, input_latency, output_latency, joint_limits, torque_limits, cmd_clip)

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


class ActuatorNetManager(object):
    def __init__(self, actuator_type, model_path=None, device='cpu', share_memory=True):
        # 'cuda' is not advised in modern cpus
        if actuator_type == 'history':
            self.actuator_class = ActuatorNetWithHistory
            self.actuator_sim_class = ActuatorNetWithHistorySim
            if not model_path:
                model_path = os.path.join(burl.rsc_path, 'actuator_net_with_history.pt')
        elif actuator_type == 'single':
            self.actuator_class = ActuatorNet
            self.actuator_sim_class = ActuatorNetSim
            if not model_path:
                model_path = os.path.join(burl.rsc_path, 'actuator_net.pt')
        else:
            raise RuntimeError(f'Unknown actuator type `{actuator_type}`')

        model_info = torch.load(model_path, map_location={'cuda:0': device})
        self._net = self.actuator_class(hidden_dims=model_info['hidden_dims']).to(device)
        self._net.load_state_dict(model_info['model'])
        if share_memory:
            self._net.share_memory()

    @property
    def make_motor(self):
        return make_part(self.actuator_sim_class, self._net)


if __name__ == '__main__':
    pass
