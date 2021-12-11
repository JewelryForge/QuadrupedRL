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


def end_trajectory_generator(k, h=0.08):  # 0.2 in paper
    if k < 0:
        return -end_trajectory_generator(-k) * 0.2
    k2 = k ** 2
    k3 = k2 * k
    if 0 < k <= 1:
        return h * (-2 * k3 + 3 * k2)
    if 1 < k < 2:
        return h * (2 * k3 - 9 * k2 + 12 * k - 4)
    return 0


def designed_tg(c=0.2, h=0.08):
    def _power(x, exp):
        _x = 1
        for _ in range(exp + 1):
            yield _x
            _x *= x

    def _poly5th(x, coeff):
        return np.asarray(coeff) @ list(_power(x, 5))

    def _solve5th(d1, d2):
        x1, y1, v1, a1 = d1
        x2, y2, v2, a2 = d2
        coeff = np.array(((1, 1, 1, 1, 1, 1),
                          (1, 1, 1, 1, 1, 1),
                          (0, 1, 2, 3, 4, 5),
                          (0, 1, 2, 3, 4, 5),
                          (0, 0, 2, 6, 12, 20),
                          (0, 0, 2, 6, 12, 20)))
        x1_pow = list(_power(x1, 5))
        x2_pow = list(_power(x2, 5))
        X = np.array((x1_pow,
                      x2_pow,
                      [0, *x1_pow[:-1]],
                      [0, *x2_pow[:-1]],
                      [0, 0, *x1_pow[:-2]],
                      [0, 0, *x2_pow[:-2]]))
        Y = np.array((y1, y2, v1, v2, a1, a2))
        print(X, coeff * X, Y, sep='\n', end='\n\n')
        return np.linalg.solve(coeff * X, Y)

    dots = np.array(((-2, 0, 0, 0),
                     (-1, -c * h, 0, 0),
                     (0, 0, 0, 0),
                     (1, h, 0, 0),
                     (2, 0, 0, 0)))
    coeffs = np.array([_solve5th(dot1, dot2) for dot1, dot2 in zip(dots, dots[1:])])

    def _tg(k):
        if k < -2 or k > 2:
            return 0.0
        return _poly5th(k, coeffs[min(3, int(k + 2))])

    return _tg


if __name__ == '__main__':
    # print(solve_5th((-2, 0, 0, 0), (-1, -0.2 * 0.08, 0, 0)))
    import matplotlib.pyplot as plt

    tg = designed_tg()
    x = np.linspace(-4, 4, 1000)
    # print(tg(0.1))
    y = [tg(x) for x in x]
    plt.plot(x, y)
    plt.show()
