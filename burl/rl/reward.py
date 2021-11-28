from abc import ABC

import numpy as np

from burl.utils.transforms import Rpy


class Reward(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def reward_reshape(lower, range_, symmetric=False):
    w = np.sqrt(np.arctanh(0.95)) / range_

    def _reward_reshape(v):
        return np.tanh(np.power((v - lower) * w, 2)) if v > lower else 0

    def _reward_reshape_symmetric(v):
        return np.tanh(np.power((v - lower) * w, 2))

    return _reward_reshape_symmetric if symmetric else _reward_reshape


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


class LinearVelocityReward(Reward):
    def __init__(self, lower=-0.1, upper=0.6):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, cmd, linear):
        projected_velocity = np.dot(cmd[:2], linear[:2])
        return self.reshape(projected_velocity)


class AngularVelocityReward(Reward):
    def __init__(self, lower=0., upper=0.6):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, cmd, angular):
        if cmd[2] != 0.0:
            projected_angular = angular[2] * cmd[2]
            return self.reshape(projected_angular)
        else:
            return 1 - self.reshape(abs(angular[2]))


class RedundantLinearPenalty(Reward):
    def __init__(self, linear_upper=0.3):
        self.reshape = reward_reshape(0, linear_upper)

    def __call__(self, cmd, linear):
        v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
        return -self.reshape(np.linalg.norm(v_o))


class RedundantAngularPenalty(Reward):
    def __init__(self, angular_upper=1.5):
        self.reshape = reward_reshape(0, angular_upper)

    def __call__(self, angular):
        w_xy = np.linalg.norm(angular[:2])
        return -self.reshape(w_xy)


class BodyPosturePenalty(Reward):
    def __init__(self, roll_upper=np.pi / 12, pitch_upper=np.pi / 6):
        self.roll_reshape = reward_reshape(0, roll_upper, symmetric=True)
        self.pitch_reshape = reward_reshape(0, pitch_upper, symmetric=True)

    def __call__(self, orientation):
        r, p, _ = Rpy.from_quaternion(orientation)
        return -(self.roll_reshape(r) + self.pitch_reshape(p)) * 0.5


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


class BodyHeightReward(Reward):
    def __init__(self, lower=0.15, upper=0.3):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, h):
        return self.reshape(h)


class TargetMutationPenalty(Reward):
    def __init__(self, upper=500):
        self.reshape = reward_reshape(0.0, upper)

    def __call__(self, smoothness):
        return -self.reshape(smoothness)


class FootClearanceReward(Reward):
    pass


class BodyCollisionPenalty(Reward):
    def __init__(self):
        pass

    def __call__(self, contact_states):
        contact_states = list(contact_states)
        for i in range(1, 5):
            contact_states[i * 3] = False
        return -sum(contact_states)


class TorquePenalty(Reward):
    def __init__(self, upper=100):
        self.reshape = reward_reshape(0, upper)

    def __call__(self, torques):
        torque_sum = sum(abs(t) for t in torques)
        return -self.reshape(torque_sum)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pass
