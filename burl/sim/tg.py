import random

import numpy as np

from burl.utils import ang_norm, g_cfg


def joint_trajectory_generator():
    pass


def power(x, exp):
    _x = 1
    for _ in range(exp + 1):
        yield _x
        _x *= x


def vertical_tg(h=0.08):  # 0.2 in paper
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


# def vertical_and_lateral_tg(h=0.12, coeff=0.2):
#     coeff1 = np.array((0., 0., 3., -2.)) * h
#     coeff2 = np.array((-4., 12., -9., 2.)) * h
#
#     def _tg(phi):
#         k = phi * 2 / np.pi
#         if k <= 0 or k >= 2:
#             return -_tg(-k) * 0.2
#         k_pow = list(power(k, 3))
#         if 0 < k <= 1:
#             z = coeff1 @ k_pow
#         if 1 < k < 2:
#             z = coeff2 @ k_pow
#         return np.array(z * coeff, 0., z)
#
#     return _tg


# def designed_tg(c=0.3, h=0.1):
#     def _poly5th(x, coeff):
#         return np.asarray(coeff) @ list(power(x, 5))
#
#     def _solve5th(d1, d2):
#         x1, y1, v1, a1 = d1
#         x2, y2, v2, a2 = d2
#         coeff = np.array(((1, 1, 1, 1, 1, 1),
#                           (1, 1, 1, 1, 1, 1),
#                           (0, 1, 2, 3, 4, 5),
#                           (0, 1, 2, 3, 4, 5),
#                           (0, 0, 2, 6, 12, 20),
#                           (0, 0, 2, 6, 12, 20)))
#         x1_pow = list(power(x1, 5))
#         x2_pow = list(power(x2, 5))
#         X = np.array((x1_pow,
#                       x2_pow,
#                       [0, *x1_pow[:-1]],
#                       [0, *x2_pow[:-1]],
#                       [0, 0, *x1_pow[:-2]],
#                       [0, 0, *x2_pow[:-2]]))
#         Y = np.array((y1, y2, v1, v2, a1, a2))
#         return np.linalg.solve(coeff * X, Y)
#
#     dots = np.array(((-2, 0, 0, 0.0),
#                      (-1, -c * h, 0, 0),
#                      (0, 0, 0.05, 0.15),
#                      (1, h, 0, -0.3),
#                      (2, 0, 0, 0.0)))
#     coeffs = np.array([_solve5th(dot1, dot2) for dot1, dot2 in zip(dots, dots[1:])])
#     num_sections = len(coeffs)
#
#     def _tg(k):
#         if k < -2 or k > 2:
#             return 0.0
#         return _poly5th(k, coeffs[min(num_sections - 1, int(k + 2))])
#
#     return _tg

class PhaseRoller(object):
    INIT_PHASES = (0.0, -np.pi, -np.pi, 0.0)
    base_frequency = 2.0  # TODO: COMPLETE THE STATE MACHINE

    def __init__(self, time_step):
        self._time_step = time_step
        self._init_phases()
        self._frequency = np.ones(4) * self.base_frequency
        self._cycles = np.zeros(4)

    @staticmethod
    def symmetric(phases):
        fr, fl, rr, rl = phases
        return fl, fr, rl, rr

    def random_symmetric(self, phases, threshold=0.5):
        return phases if random.random() > threshold else self.symmetric(phases)

    def _init_phases(self):
        if g_cfg.tg_init == 'fixed':
            self.__init_phases = np.array(self.INIT_PHASES, dtype=np.float32)
        elif g_cfg.tg_init == 'symmetric':
            self.__init_phases = np.array(self.random_symmetric(self.INIT_PHASES), dtype=np.float32)
        elif g_cfg.tg_init == 'random':
            self.__init_phases = ang_norm(np.random.random(4) * 2 * np.pi).astype(np.float32)
        else:
            raise RuntimeError(f'Unknown TG Init Mode {g_cfg.tg_init}')
        self._phases = self.__init_phases.copy()

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
        self._init_phases()
        self._frequency = np.ones(4) * self.base_frequency
        self._cycles = np.zeros(4)

    def update(self):
        _phases = self._phases.copy()
        self._phases += self._frequency * self._time_step * 2 * np.pi
        flags = np.logical_and(ang_norm(_phases - self.__init_phases) < 0,
                               ang_norm(self._phases - self.__init_phases) >= 0)
        self._cycles[flags.nonzero(),] += 1
        self._phases = ang_norm(self._phases)
        return self._phases


class TgStateMachine(PhaseRoller):
    def __init__(self, time_step, make_tg, lower_frequency=0.5, upper_frequency=3.0):
        # We set the base frequency f0 to zero when the zero command is given for 0.5s,
        # which stops FTGs, and the robot stands still on the terrain.
        # f0 is set to 1.25 Hz when the direction command is given
        # or the linear velocity of the base exceeds 0.3 m/s for the disturbance rejection.
        # The state machine is included in the training environment.
        super().__init__(time_step)
        self._lower_frequency, self._upper_frequency = lower_frequency, upper_frequency
        self._tg = make_tg()

    def update(self, frequency_offsets):
        frequency_offsets = np.asarray(frequency_offsets)
        self._frequency = self.base_frequency + frequency_offsets
        self._frequency = np.clip(self._frequency, self._lower_frequency, self._upper_frequency)
        return super().update()

    def get_priori_trajectory(self):
        return np.asarray(self._tg(self._phases))
        # k: [-2, 2)
        # return np.array([(phi) for phi in self._phases])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # g_cfg.tg_init = 'symmetric'
    # stm = LocomotionStateMachine(0.01)
    # print(stm.phases)
    # for _ in range(10):
    #     stm._init_phases()
    #     print(stm.phases)

    # tg = designed_tg()
    # tg2 = end_trajectory_generator()
    x = np.linspace(-2, 2, 1000)
    y1 = [tg(x) for x in x]
    # y2 = [tg2(x) for x in x]
    plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.legend(['des', 'raw'])
    plt.show()
