import math
from abc import ABC

import numpy as np


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


def tanh_reshape(lower, upper):
    """
    Zoom tanh so that f(lower) = -0.9 and f(upper) = 0.9
    :return: function f
    """
    middle = (lower + upper) / 2
    w = np.arctanh(0.9) / (upper - middle)

    def _reshape(v):
        return np.tanh((v - middle) * w)

    return _reshape


def elu_reshape(coeff):
    def _reshape(v):
        return (v if v >= 0 else math.exp(v) - 1) * coeff

    return _reshape


class LinearVelocityReward(Reward):
    def __init__(self, lower=-0.1, upper=0.6):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, cmd, linear):
        projected_velocity = np.dot(cmd[:2], linear[:2])
        return self.reshape(projected_velocity)


class EluLinearVelocityReward(Reward):
    def __init__(self, coeff=2.0):
        self.reshape = elu_reshape(coeff)

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


class YawRateReward(Reward):
    def __init__(self, upper=0.6):
        self.reshape = reward_reshape(0.0, upper)

    def __call__(self, yaw_cmd, yaw_rate):
        if yaw_cmd != 0.0:
            return self.reshape(yaw_rate * yaw_cmd)
        else:
            return -self.reshape(abs(yaw_rate))


class RollPitchRatePenalty(Reward):
    def __init__(self, dr_upper=3.0, dp_upper=1.5):
        self.dr_reshape = reward_reshape(0.0, dr_upper)
        self.dp_reshape = reward_reshape(0.0, dp_upper)

    def __call__(self, r_rate, p_rate):
        # print(self.dr_reshape(r_rate), self.dr_reshape(p_rate))
        return -(self.dr_reshape(r_rate) + self.dp_reshape(p_rate)) / 2


class RedundantLinearPenalty(Reward):
    def __init__(self, linear_upper=0.3):
        self.reshape = reward_reshape(0, linear_upper)

    def __call__(self, cmd, linear):
        v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
        return 1 - self.reshape(np.linalg.norm(v_o))


class RedundantAngularPenalty(Reward):
    def __init__(self, angular_upper=1.5):
        self.reshape = reward_reshape(0, angular_upper)

    def __call__(self, angular):
        w_xy = np.linalg.norm(angular[:2])
        return 1 - self.reshape(w_xy)


class BodyPosturePenalty(Reward):
    def __init__(self, roll_upper=np.pi / 12, pitch_upper=np.pi / 6):
        self.roll_reshape = reward_reshape(0, roll_upper, symmetric=True)
        self.pitch_reshape = reward_reshape(0, pitch_upper, symmetric=True)

    def __call__(self, r, p):
        return 1 - (self.roll_reshape(r) + self.pitch_reshape(p)) / 2


# class BaseStabilityReward(Reward):
#     def __init__(self, linear_upper=0.3, angular_upper=1.5):
#         self.reshape_linear = reward_reshape(0, linear_upper)
#         self.reshape_angular = reward_reshape(0, angular_upper)
#
#     def __call__(self, cmd, linear, angular):
#         v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
#         v_o = np.linalg.norm(v_o)
#         w_xy = angular[:2]
#         w_xy = np.linalg.norm(w_xy)
#         return (1 - self.reshape_linear(v_o)) + (1 - self.reshape_angular(w_xy))


class BodyHeightReward(Reward):
    def __init__(self, lower=0.2, upper=0.4):
        self.reshape = reward_reshape(lower, upper - lower)

    def __call__(self, h):
        return self.reshape(h)


class TargetMutationPenalty(Reward):
    def __init__(self, upper=500):
        self.reshape = reward_reshape(0.0, upper)

    def __call__(self, smoothness):
        return 1 - self.reshape(smoothness)


class FootSlipPenalty(Reward):
    def __init__(self, lower=0.1, upper=0.3):
        # Due to the error of slip velocity estimation, tolerate error of 0.1
        self.reshape = reward_reshape(lower, upper)

    def __call__(self, slip):
        return 1 - self.reshape(slip)


class SmallStridePenalty(Reward):
    def __init__(self, lower=-0.2, upper=0.4):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, strides):
        return sum(self.reshape(s) for s in strides if s != 0.0)
        # return 1 - sum(1 - self.reshape(s) for s in strides if s != 0.0)


class FootClearanceReward(Reward):
    pass


class BodyCollisionPenalty(Reward):
    def __init__(self):
        pass

    def __call__(self, contact_states):
        contact_states = list(contact_states)
        for i in range(1, 5):
            contact_states[i * 3] = False
        return 1 - sum(contact_states)


class TorquePenalty(Reward):
    def __init__(self, lower=50, upper=150):
        self.reshape = reward_reshape(lower, upper)

    def __call__(self, torques):
        torque_sum = sum(abs(t) for t in torques)
        return 1 - self.reshape(torque_sum)


class CostOfTransportReward(Reward):
    def __init__(self, lower=0.5, upper=2.5):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, cot):
        return -self.reshape(cot)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r = tanh_reshape(-0.2, 0.4)
    x = np.linspace(-0.3, 0.5, 100)
    plt.plot(x, r(x))
    plt.show()
    print(r(0.0))
    print(r(0.1))
    print(r(0.2))
    print(r(0.3))
    print(r(0.4))
