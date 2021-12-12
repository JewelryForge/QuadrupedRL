import numpy as np

from burl.utils import normalize


class LocomotionStateMachine(object):
    FIXED_INIT = True
    # INIT_ANGLES = (0, 1.5, 0.5, 1.0)
    INIT_ANGLES = (0.0, 1.0, 1.0, 0.0)

    def __init__(self, time_step, **kwargs):
        # We set the base frequency f0 to zero when the zero command is given for 0.5s,
        # which stops FTGs, and the robot stands still on the terrain.
        # f0 is set to 1.25 Hz when the direction command is given
        # or the linear velocity of the base exceeds 0.3 m/s for the disturbance rejection.
        # The state machine is included in the training environment.
        self._time_step = time_step
        if self.FIXED_INIT:
            self._phases = normalize(np.array(self.INIT_ANGLES) * np.pi)
        else:
            self._phases = normalize(np.random.random(4) * 2 * np.pi)
        self._frequency = np.ones(4) * self.base_frequency
        self._lower_frequency = kwargs.get('lower_frequency', 0.5)
        self._upper_frequency = kwargs.get('upper_frequency', 3.0)
        self._flags = np.zeros(4, dtype=bool)
        self._tg = designed_tg()
        # self._tg = end_trajectory_generator()

    @property
    def base_frequency(self):
        return 1.25  # TODO: COMPLETE THE STATE MACHINE

    @property
    def phases(self):
        return self._phases

    @property
    def frequency(self):
        return self._frequency

    @property
    def flags(self):
        return self._flags

    def reset(self):
        if self.FIXED_INIT:
            self._phases = normalize(np.array(self.INIT_ANGLES) * np.pi)
        else:
            self._phases = normalize(np.random.random(4) * 2 * np.pi)

    def update(self, frequency_offsets):
        frequency_offsets = np.asarray(frequency_offsets)
        self._frequency = self.base_frequency + frequency_offsets
        self._frequency = np.clip(self._frequency, self._lower_frequency, self._upper_frequency)
        self._phases += self._frequency * self._time_step * 2 * np.pi
        phases = normalize(self._phases)  # [-pi, pi)
        # print(self._frequency, self._phases)
        self._flags = (phases == self._phases)
        self._phases = phases
        return self._phases

    def get_priori_trajectory(self):
        # k: [-2, 2)
        return np.array([self._tg(phi / np.pi * 2) for phi in self._phases])


def joint_trajectory_generator():
    pass


def power(x, exp):
    _x = 1
    for _ in range(exp + 1):
        yield _x
        _x *= x


def end_trajectory_generator(h=0.08):  # 0.2 in paper
    coeff1 = np.array((0., 0., 3., -2.)) * h
    coeff2 = np.array((-4., 12., -9., 2.)) * h

    def _tg(k):
        if k < 0:
            return -_tg(-k) * 0.2
        k_pow = list(power(k, 3))
        if 0 < k <= 1:
            return coeff1 @ k_pow
        if 1 < k < 2:
            return coeff2 @ k_pow
        return 0

    return _tg


def designed_tg(c=0.2, h=0.08):
    def _poly5th(x, coeff):
        return np.asarray(coeff) @ list(power(x, 5))

    def _solve5th(d1, d2):
        x1, y1, v1, a1 = d1
        x2, y2, v2, a2 = d2
        coeff = np.array(((1, 1, 1, 1, 1, 1),
                          (1, 1, 1, 1, 1, 1),
                          (0, 1, 2, 3, 4, 5),
                          (0, 1, 2, 3, 4, 5),
                          (0, 0, 2, 6, 12, 20),
                          (0, 0, 2, 6, 12, 20)))
        x1_pow = list(power(x1, 5))
        x2_pow = list(power(x2, 5))
        X = np.array((x1_pow,
                      x2_pow,
                      [0, *x1_pow[:-1]],
                      [0, *x2_pow[:-1]],
                      [0, 0, *x1_pow[:-2]],
                      [0, 0, *x2_pow[:-2]]))
        Y = np.array((y1, y2, v1, v2, a1, a2))
        return np.linalg.solve(coeff * X, Y)

    dots = np.array(((-2, 0, 0, 0.0),
                     (-1, -c * h, 0, 0),
                     (0, 0, 0.05, 0.15),
                     (1, h, 0, -0.3),
                     (2, 0, 0, 0.0)))
    coeffs = np.array([_solve5th(dot1, dot2) for dot1, dot2 in zip(dots, dots[1:])])
    num_sections = len(coeffs)

    def _tg(k):
        if k < -2 or k > 2:
            return 0.0
        return _poly5th(k, coeffs[min(num_sections - 1, int(k + 2))])

    return _tg


if __name__ == '__main__':
    # print(solve_5th((-2, 0, 0, 0), (-1, -0.2 * 0.08, 0, 0)))
    import matplotlib.pyplot as plt

    tg = designed_tg()
    tg2 = end_trajectory_generator()
    x = np.linspace(-2, 2, 1000)
    # print(tg(0.1))
    # y1 = [tg(x) for x in x]
    y2 = [tg2(x) for x in x]
    # plt.plot(x, y1)
    plt.plot(x, y2)
    # plt.legend(['des', 'raw'])
    plt.show()
