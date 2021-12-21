import math
from abc import ABC

import numpy as np


class Reward(ABC):
    def __call__(self, cmd, env, robot):
        raise NotImplementedError


def tanh2_reshape(lower, upper):
    w = np.sqrt(np.arctanh(0.95)) / (upper - lower)

    def _reshape(v):
        return np.tanh(np.power((v - lower) * w, 2)) if v > lower else 0

    return _reshape


def tanh2_reverse(lower, upper, value):
    w = np.sqrt(np.arctanh(0.95)) / (upper - lower)
    return np.sqrt(np.arctanh(value)) / w + lower


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


def tanh_reverse(lower, upper, value):
    middle = (lower + upper) / 2
    w = np.arctanh(0.9) / (upper - middle)
    return np.arctanh(value) / w + middle


def elu_reshape(coeff):
    def _reshape(v):
        return (v if v >= 0 else math.exp(v) - 1) * coeff

    return _reshape


class LinearVelocityReward(Reward):
    def __init__(self, lower=-0.15, upper=0.45):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        linear = robot.getBaseLinearVelocityInBaseFrame()
        projected_velocity = np.dot(cmd[:2], linear[:2])
        return self.reshape(projected_velocity)


class EluLinearVelocityReward(Reward):
    def __init__(self, coeff=2.0):
        self.reshape = elu_reshape(coeff)

    def __call__(self, cmd, env, robot):
        linear = robot.getBaseLinearVelocityInBaseFrame()
        projected_velocity = np.dot(cmd[:2], linear[:2])
        return self.reshape(projected_velocity)


# class AngularVelocityReward(Reward):
#     def __init__(self, lower=0., upper=0.6):
#         self.reshape = tanh2_reshape(lower, upper)
#
#     def __call__(self, cmd, env, robot):
#         angular = robot.getBaseAngularVelocity()
#         if cmd[2] != 0.0:
#             projected_angular = angular[2] * cmd[2]
#             return self.reshape(projected_angular)
#         else:
#             return 1 - self.reshape(abs(angular[2]))


class YawRateReward(Reward):
    def __init__(self, upper_pos=0.6, upper_neg=0.6):
        self.reshape_pos = tanh2_reshape(0.0, upper_pos)
        self.reshape_neg = tanh2_reshape(0.0, upper_neg)

    def __call__(self, cmd, env, robot):
        yaw_cmd, yaw_rate = cmd[2], robot.getBaseRpyRate()[2]
        if yaw_cmd != 0.0:
            return self.reshape_pos(yaw_rate * yaw_cmd)
        else:
            return -self.reshape_neg(abs(yaw_rate))


class RollPitchRatePenalty(Reward):
    def __init__(self, dr_upper=1.0, dp_upper=0.5):
        self.dr_reshape = tanh2_reshape(0.0, dr_upper)
        self.dp_reshape = tanh2_reshape(0.0, dp_upper)

    def __call__(self, cmd, env, robot):
        r_rate, p_rate, _ = robot.getBaseRpyRate()
        return -(self.dr_reshape(r_rate) + self.dp_reshape(p_rate)) / 2


class RedundantLinearPenalty(Reward):
    def __init__(self, linear_upper=0.3):
        self.reshape = tanh2_reshape(0, linear_upper)

    def __call__(self, cmd, env, robot):
        linear = robot.getBaseLinearVelocityInBaseFrame()[:2]
        v_o = np.asarray(linear) - np.asarray(cmd[:2]) * np.dot(linear, cmd[:2])
        return 1 - self.reshape(np.linalg.norm(v_o))


# class RedundantAngularPenalty(Reward):
#     def __init__(self, angular_upper=1.5):
#         self.reshape = tanh2_reshape(0, angular_upper)
#
#     def __call__(self, cmd, env, robot):
#         angular = robot.getBaseAngularVelocity()
#         w_xy = np.linalg.norm(angular[:2])
#         return 1 - self.reshape(w_xy)


class BodyPosturePenalty(Reward):
    def __init__(self, roll_upper=np.pi / 12, pitch_upper=np.pi / 6):
        self.roll_reshape = tanh2_reshape(0, roll_upper)
        self.pitch_reshape = tanh2_reshape(0, pitch_upper)

    def __call__(self, cmd, env, robot):
        r, p, _ = env.getSafetyRpyOfRobot()
        return 1 - (self.roll_reshape(abs(r)) + self.pitch_reshape(abs(p))) / 2


# class BaseStabilityReward(Reward):
#     def __init__(self, linear_upper=0.3, angular_upper=1.5):
#         self.reshape_linear = tanh2_reshape(0, linear_upper)
#         self.reshape_angular = tanh2_reshape(0, angular_upper)
#
#     def __call__(self, cmd, linear, angular):
#         v_o = np.asarray(linear[:2]) - np.asarray(cmd[:2]) * np.dot(linear[:2], cmd[:2])
#         v_o = np.linalg.norm(v_o)
#         w_xy = angular[:2]
#         w_xy = np.linalg.norm(w_xy)
#         return (1 - self.reshape_linear(v_o)) + (1 - self.reshape_angular(w_xy))


class BodyHeightReward(Reward):
    def __init__(self, lower=0.2, upper=0.35):
        self.reshape = tanh2_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        height = env.getSafetyHeightOfRobot()
        return self.reshape(height)


class TargetMutationPenalty(Reward):
    def __init__(self, upper=500):
        self.reshape = tanh2_reshape(0.0, upper)

    def __call__(self, cmd, env, robot):
        mutation = env.getActionMutation()
        return 1 - self.reshape(mutation)


class FootSlipPenalty(Reward):
    def __init__(self, lower=0.2, upper=1.0):
        # Due to the error of slip velocity estimation, tolerate error of 0.2
        self.reshape = tanh2_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        slips = robot.getFootSlipVelocity()
        return -sum(self.reshape(s) for s in slips)


class SmallStridePenalty(Reward):
    def __init__(self, lower=-0.2, upper=0.4):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        strides = [np.dot(s, cmd[:2]) for s in robot.getStrides()]
        return sum(self.reshape(s) for s in strides if s != 0.0)
        # return 1 - sum(1 - self.reshape(s) for s in strides if s != 0.0)


class FootClearanceReward(Reward):
    pass


class BodyCollisionPenalty(Reward):
    def __init__(self):
        pass

    def __call__(self, cmd, env, robot):
        contact_states = list(robot.getContactStates())
        for i in range(1, 5):
            contact_states[i * 3] = False
        return 1 - sum(contact_states)


class TorquePenalty(Reward):
    def __init__(self, lower=50, upper=150):
        self.reshape = tanh2_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        torque_sum = sum(abs(t) for t in robot.getLastAppliedTorques())
        return 1 - self.reshape(torque_sum)


class CostOfTransportReward(Reward):
    def __init__(self, lower=0.0, upper=2.0):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        cot = robot.getCostOfTransport()
        return -self.reshape(cot)


class RewardRegistry(object):
    def __init__(self, cmd, env, robot):
        self._cmd, self._env, self._robot = cmd, env, robot
        self._rewards_set = set()
        self._rewards_weights = []
        self._weight_sum = 0.0
        self._coefficient = 1.0
        self._reward_details = {}

    @property
    def weight_sum(self):
        return self._weight_sum

    def register(self, name: str, weight: float):
        if name in self._rewards_set:
            raise RuntimeError(f'Duplicated Reward Type {name}')
        self._rewards_set.add(name)
        reward_class = eval(name)
        self._weight_sum += weight
        self._rewards_weights.append((reward_class(), weight))

    def setCoefficient(self, coeff):
        self._coefficient = coeff

    def report(self):
        from burl.utils.log import colored_str
        print(colored_str(f'Got {len(self._rewards_weights)} types of rewards:', 'white'))
        print(f"{'Reward Type':<25}Weight * {self._coefficient:.3f}")
        for reward, weight in self._rewards_weights:
            reward_name: str = reward.__class__.__name__
            length = len(reward_name)
            if reward_name.endswith('Reward'):
                reward_name = colored_str(reward_name, 'green')
            elif reward_name.endswith('Penalty'):
                reward_name = colored_str(reward_name, 'magenta')
            print(f'{reward_name}{" " * (25 - length)}{weight:.3f}')
        print()

    def getRewardDetails(self):
        return self._reward_details

    def calculateReward(self):
        self._reward_details.clear()
        weighted_sum = 0.0
        for reward, weight in self._rewards_weights:
            rew = reward(self._cmd, self._env, self._robot)
            self._reward_details[reward.__class__.__name__] = rew
            weighted_sum += rew * weight
        return weighted_sum * self._coefficient


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    registry = RewardRegistry(1, 2, 3)
    registry.register('CostOfTransportReward', 0.1)
    # registry.register('RedundantAngularPenalty', 0.2)
    registry.report()

    print(tanh_reverse(-0.15, 0.45, 0))
    # r1 = tanh2_reshape(0.0, 2.0)
    # r = tanh_reshape(0.0, 2.0)
    # x = np.linspace(-0.5, 2.5, 1000)
    # plt.plot(x, [r1(x) for x in x])
    # plt.plot(x, [(r(x) + 1) / 2 for x in x])
    # plt.show()
