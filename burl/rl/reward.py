from abc import ABC

import numpy as np


class Reward(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def reward_reshape(lower, _range):
    w = np.sqrt(np.arctanh(0.95)) / _range

    def _reward_reshape(v):
        return np.tanh(np.power((v - lower) * w, 2)) if v > lower else 0

    return _reward_reshape


# class LinearVelocityTruncatedReward(Reward):
#     def __init__(self, upper=0.6, decay=2.0):
#         self._upper = upper
#         self._decay = decay
#
#     def __call__(self, cmd, linear):
#         projected_velocity = np.dot(cmd[:2], linear[:2])
#         if projected_velocity > self._upper:
#             return 1.0
#         if cmd == (0.0, 0.0):
#             return 0.0
#         return np.exp(-self._decay * (projected_velocity - self._upper) ** 2)
#
#
# class AngularVelocityTruncatedReward(Reward):
#     def __init__(self, upper=0.6, decay=1.5):
#         self._upper = upper
#         self._decay = decay
#
#     def __call__(self, cmd, angular):
#         if cmd[2] != 0.0:
#             projected_angular = angular[2] * cmd[2]
#             if projected_angular >= self._upper:
#                 return 1.0
#             return np.exp(-self._decay * (projected_angular - self._upper) ** 2)
#         else:
#             return np.exp(-self._decay * angular[2] ** 2)
#
#
# class BaseStabilityReward(Reward):
#     def __init__(self, decay_linear=1.5, decay_angular=1.5):
#         self._decay_linear = decay_linear
#         self._decay_angular = decay_angular  # TODO: Try Truncate
#
#     def __call__(self, cmd, linear, angular):
#         v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
#         w_xy = angular[:2]
#         return np.exp(-self._decay_linear * np.dot(v_o, v_o)) + np.exp(-self._decay_angular * np.dot(w_xy, w_xy))



class LinearVelocityTruncatedReward(Reward):
    def __init__(self, lower=-0.1, upper=0.6):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, cmd, linear):
        projected_velocity = np.dot(cmd[:2], linear[:2])
        return self.reshape(projected_velocity)


class AngularVelocityTruncatedReward(Reward):
    def __init__(self, lower=0., upper=0.6):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, cmd, angular):
        if cmd[2] != 0.0:
            projected_angular = angular[2] * cmd[2]
            return self.reshape(projected_angular)
        else:
            return 1 - self.reshape(abs(angular[2]))


class BaseStabilityReward(Reward):
    def __init__(self, linear_upper=0.3, angular_upper=1.5):
        self.reshape_linear = reward_reshape(0, linear_upper)
        self.reshape_angular = reward_reshape(0, angular_upper)

    def __call__(self, cmd, linear, angular):
        v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
        v_o = np.linalg.norm(v_o)
        w_xy = angular[:2]
        w_xy = np.linalg.norm(w_xy)
        return (1 - self.reshape_linear(v_o)) + (1 - self.reshape_angular(w_xy))


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
    import matplotlib.pyplot as plt
    pass

