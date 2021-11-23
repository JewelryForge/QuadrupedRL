from burl.rl.reward import *
from burl.sim.quadruped import Quadruped


class BasicTask(object):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        self._env = env
        self._robot: Quadruped = env.robot
        self._rewards = (
            LinearVelocityTruncatedReward(),
            AngularVelocityTruncatedReward(),
            BaseStabilityReward(),
            BodyCollisionReward(),
            TorqueReward()
        )

        # self._weights = [
        #     0.2, 0.05, 0.01, 0.02, 2e-5
        # ]

        self._weights = (
            0.05, 0.05, 0.04, 0.02, 2e-5
        )
        self._cmd = cmd

    @property
    def cmd(self):
        return self._cmd

    def calculateReward(self):
        linear = self._robot.getBaseLinearVelocityInBaseFrame()
        angular = self._robot.getBaseAngularVelocityInBaseFrame()
        contact_states = self._robot.getContactStates()
        torques = self._robot.getLastAppliedTorques()
        rewards = (
            self._rewards[0](self._cmd, linear),
            self._rewards[1](self._cmd, angular),
            self._rewards[2](self._cmd, linear, angular),
            self._rewards[3](contact_states),
            self._rewards[4](torques)
         )
        weighted_rewards = [r * w for r, w in zip(rewards, self._weights)]
        # print(np.array(weighted_rewards))
        # print(np.array(linear), np.array(angular), np.array(weighted_rewards))
        return sum(weighted_rewards)

    def reset(self):
        pass

    def done(self):
        return False
