import gym
from gym.utils import seeding

from terrain import make_plain
from quadruped import QuadrupedSim


class LocomotionEnv(gym.Env):
    def __init__(self, make_robot, **kwargs):
        super(LocomotionEnv, self).__init__()
        self._robot: QuadrupedSim = make_robot()
        self._terrain_generator = kwargs.get('terrain_generator', make_plain)

        self._random_state, self._random_seed = self.seed(kwargs.get('seed', None))
        self._sim_frequency = kwargs.get('sim_frequency', 240)
        self._action_frequency = kwargs.get('action_frequency', 240)
        assert self._sim_frequency >= self._robot.frequency >= self._action_frequency

        self._num_action_repeats = int(self._sim_frequency / self._action_frequency)
        self._num_execution_repeats = int(self._sim_frequency / self._robot.frequency)
        # self._num_action_repeats = self._action_time_step / self

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    LocomotionEnv(QuadrupedSim)
