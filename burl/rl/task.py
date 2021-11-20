from burl.rl.reward import *


class SimpleForwardTaskOnFlat(object):
    """Default empty task."""

    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        """Initializes the task."""
        self._env = env
        self._rewards = [
            LinearVelocityTruncatedReward(),
            AngularVelocityTruncatedReward(),
            BaseStabilityReward(),
            BodyCollisionReward(),
            TorqueReward()
        ]
        self._weights = [
            0.05, 0.05, 0.04, 0.02, 2e-5
        ]
        self._cmd = cmd

    @property
    def cmd(self):
        return self._cmd

    def __call__(self, info):
        base_state = info['base_state']
        linear, angular = base_state.twist.linear, base_state.twist.angular
        rewards = [
            self._rewards[0](self._cmd, linear),
            self._rewards[1](self._cmd, angular),
            self._rewards[2](self._cmd, linear, angular),
            self._rewards[3](info['contact_states']),
            self._rewards[4](info['torques'])
        ]
        weighted_rewards = [r * w for r, w in zip(rewards, self._weights)]
        # print(weighted_rewards)
        return sum(weighted_rewards)

    def reset(self):
        pass

    def done(self):
        """Checks if the episode is over.

           If the robot base becomes unstable (based on orientation), the episode
           terminates early.
        """
        return False
