import numpy as np
from burl.utils import normalize, M_2_PI


class LocomotionStateMachine(object):
    def __init__(self, time_step):
        # We set the base frequency f 0 to zero when the zero command is given for 0.5 s,
        # which stops FTGs, and the robot stands still on the terrain.
        # f0 is set to 1.25 Hz when the direction command is given
        # or the linear velocity of the base exceeds 0.3 m/s for the disturbance rejection.
        # The state machine is included in the training environment.
        self._time_step = time_step
        self._phases = normalize(np.random.random(4) * M_2_PI)

    @property
    def base_frequency(self):
        return 1.25  # TODO: COMPLETE THE STATE MACHINE

    @property
    def phases(self):
        return self._phases

    def update(self, frequency_offsets):
        frequency_offsets = np.asarray(frequency_offsets)
        self._phases += (self.base_frequency + frequency_offsets) * self._time_step * M_2_PI
        self._phases = normalize(self._phases)  # [-pi, pi)

    def priori_trajectory(self):
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
    lsm = LocomotionStateMachine(0.1)
    np.set_printoptions(3)
    for _ in range(100):
        lsm.update((0., 0., 0., 0.))
        print(lsm.priori_trajectory())
        # print(lsm._phases / M_2_PI)
