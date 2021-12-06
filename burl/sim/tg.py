import numpy as np

from burl.utils import normalize


class LocomotionStateMachine(object):
    FIXED_INIT = True
    # INIT_ANGLES = (0, 1.5, 0.5, 1.0)
    INIT_ANGLES = (0.0, 1.0, 1.0, 0.0)

    def __init__(self, time_step, **kwargs):
        # We set the base frequency f 0 to zero when the zero command is given for 0.5 s,
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

    @property
    def base_frequency(self):
        return 1.25  # TODO: COMPLETE THE STATE MACHINE

    @property
    def phases(self):
        return self._phases

    @property
    def frequency(self):
        return self._frequency

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
        self._phases = normalize(self._phases)  # [-pi, pi)
        return self._phases

    def get_priori_trajectory(self):
        # k: [-2, 2)
        return np.array([end_trajectory_generator(phi / np.pi * 2) for phi in self._phases])


def joint_trajectory_generator():
    pass


def end_trajectory_generator(k, h=0.12):  # 0.2 in paper
    k2 = k ** 2
    k3 = k2 * k
    if 0 < k <= 1:
        return h * (-2 * k3 + 3 * k2 ** 2)
    if 1 < k < 2:
        return h * (2 * k3 - 9 * k2 + 12 * k - 4)
    return 0


if __name__ == '__main__':
    pass
