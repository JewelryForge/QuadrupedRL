import enum
import random

import numpy as np

from qdpgym.utils import Angle


def power(x, exp):
    _x = 1
    for _ in range(exp + 1):
        yield _x
        _x *= x


def vertical_tg(h=0.12):  # 0.2 in paper
    coeff1 = np.array((0., 0., 3., -2.)) * h
    coeff2 = np.array((-4., 12., -9., 2.)) * h

    def _tg(phases):
        priori = []
        for phi in phases:
            k = phi * 2 / np.pi
            if k <= 0 or k >= 2:
                priori.append(np.zeros(3))
            k_pow = list(power(k, 3))
            if 0 < k <= 1:
                priori.append((0., 0., coeff1 @ k_pow))
            if 1 < k < 2:
                priori.append((0., 0., coeff2 @ k_pow))
        return np.array(priori)

    return _tg


class PhaseRoller(object):
    TROT = (0.0, -np.pi, -np.pi, 0.0)
    WALK = (0.0, -np.pi / 2, -np.pi, -np.pi * 3 / 2)
    PACE = (0.0, -np.pi, 0.0, -np.pi)
    base_frequency = 2.0  # TODO: COMPLETE THE STATE MACHINE

    class InitType(enum.IntEnum):
        FIXED = 0
        SYMMETRIC = 1
        RANDOM = 2

    def __init__(self, time_step, random_state, init_type='random'):
        self._time_step = time_step
        self._random = random_state
        self._init_type = self._check_init_type(init_type)

        self._init_phases = self._get_init_phases()
        self._phases = self._init_phases.copy()
        self._frequency = np.ones(4) * self.base_frequency
        self._cycles = np.zeros(4)

    @staticmethod
    def symmetric(phases):
        fr, fl, rr, rl = phases
        return fl, fr, rl, rr

    def random_symmetric(self, phases):
        return phases if self._random.random() > 0.5 else self.symmetric(phases)

    def _check_init_type(self, init_type):
        if init_type == 'fixed':
            return self.InitType.FIXED
        elif init_type == 'symmetric':
            return self.InitType.SYMMETRIC
        elif init_type == 'random':
            return self.InitType.RANDOM
        else:
            raise RuntimeError(f'Unknown TG Init Mode {init_type}')

    def _get_init_phases(self):
        if self._init_type == self.InitType.FIXED:
            return np.array(self.TROT, dtype=np.float32)
        elif self._init_type == self.InitType.SYMMETRIC:
            return np.array(self.random_symmetric(self.TROT), dtype=np.float32)
        elif self._init_type == self.InitType.RANDOM:
            return self._random.uniform(-np.pi, np.pi, 4).astype(np.float32)

    @property
    def phases(self):
        return self._phases

    @property
    def frequency(self):
        return self._frequency

    @property
    def cycles(self):
        return self._cycles

    def reset(self):
        self._init_phases = self._get_init_phases()
        self._phases = self._init_phases.copy()
        self._frequency = np.ones(4) * self.base_frequency
        self._cycles = np.zeros(4)

    def update(self):
        _phases = self._phases.copy()
        self._phases += self._frequency * self._time_step * 2 * np.pi
        flags = np.logical_and(Angle.norm(_phases - self._init_phases) < 0,
                               Angle.norm(self._phases - self._init_phases) >= 0)
        self._cycles[(flags.nonzero(),)] += 1
        self._phases = Angle.norm(self._phases)
        return self._phases


class TgStateMachine(PhaseRoller):
    def __init__(
        self,
        time_step,
        random_state,
        traj_gen,
        init_type='symmetric',
        freq_bounds=(0.5, 3.0)
    ):
        super().__init__(time_step, random_state, init_type)
        self._freq_bounds = freq_bounds
        self._traj_gen = traj_gen

    def update(self, frequency_offsets=None):  # FIXME: how freq_offsets works?
        if frequency_offsets is not None:
            frequency_offsets = np.asarray(frequency_offsets)
            self._frequency = self.base_frequency + frequency_offsets
            self._frequency = np.clip(self._frequency, *self._freq_bounds)
        return super().update()

    def get_priori_trajectory(self):
        return np.asarray(self._traj_gen(self._phases))
