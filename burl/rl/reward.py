import numpy as np


class Reward(object):
    pass


class LinearVelocityTruncatedReward(Reward):
    def __init__(self, upper=0.6, decay=2.0):
        self._upper = upper
        self._decay = decay

    def __call__(self, cmd, linear):
        projected_velocity = np.dot(cmd[:2], linear[:2])
        if projected_velocity > self._upper:
            return 1.0
        if cmd == (0.0, 0.0):
            return 0.0
        return np.exp(-self._decay * (projected_velocity - self._upper) ** 2)


class AngularVelocityTruncatedReward(Reward):
    def __init__(self, upper=0.6, decay=1.5):
        self._upper = upper
        self._decay = decay

    def __call__(self, cmd, angular):
        if cmd[2] != 0.0:
            projected_angular = angular[2] * cmd[2]
            if projected_angular >= self._upper:
                return 1.0
            return np.exp(-self._decay * (projected_angular - self._upper) ** 2)
        else:
            return np.exp(-self._decay * angular[2] ** 2)


class BaseStabilityReward(Reward):
    def __init__(self, decay_linear=1.5, decay_angular=1.5):
        self._decay_linear = decay_linear
        self._decay_angular = decay_angular  # TODO: Try Truncate

    def __call__(self, cmd, linear, angular):
        v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
        w_xy = angular[:2]
        return np.exp(-self._decay_linear * np.dot(v_o, v_o)) + np.exp(-self._decay_angular * np.dot(w_xy, w_xy))


class FootClearanceReward(Reward):
    pass


class BodyCollisionReward(Reward):
    def __init__(self):
        pass

    def __call__(self, contact_states):
        contact_states = list(contact_states)
        for i in range(1, 5):
            contact_states[i * 3] = False
        return -sum(contact_states)


class TorqueReward(Reward):
    def __init__(self):
        pass

    def __call__(self, torques):
        return -sum(abs(t) for t in torques)


if __name__ == '__main__':
    r = AngularVelocityTruncatedReward()
    print(r((0., 0., 1.0), 0.5))
