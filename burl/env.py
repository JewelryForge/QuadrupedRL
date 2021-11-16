import time

import numpy as np
import pybullet
from pybullet_utils import bullet_client
import gym
from gym import spaces
from gym.utils import seeding

from burl.bc import Observable
from burl.sensors import OrientationRpySensor, GravityVectorSensor
from burl.utils import make_class
from burl.terrain import make_plane
from burl.quadruped import QuadrupedSim, QuadrupedBase


def trajectory_generator(k, h=0.12):  # 0.2 in paper
    # k = 2 * normalize(phase) / np.pi
    if 0 < k <= 1:
        return h * (-2 * k ** 3 + 3 * k ** 2)
    if 1 < k <= 2:
        return h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4)
    return 0


class LocomotionEnv(gym.Env, Observable):
    def __init__(self, **kwargs):
        self._gui: bool = kwargs.get('use_gui', True)
        if True:
            self._env = bullet_client.BulletClient(connection_mode=pybullet.GUI if self._gui else pybullet.DIRECT)
        else:
            self._env = pybullet
        _make_sensors = kwargs.get('make_sensors', [])
        super().__init__(make_sensors=_make_sensors)
        _make_robot = kwargs.get('make_robot', QuadrupedSim)
        self._robot: QuadrupedBase = _make_robot(sim_env=self._env)
        self._subordinates.append(self._robot)
        self._terrain_generator = kwargs.get('terrain_generator', make_plane)
        self._random_state, self._random_seed = self.seed(kwargs.get('seed', None))
        self._sim_frequency = kwargs.get('sim_frequency', 240)
        self._action_frequency = kwargs.get('action_frequency', 240)
        # self.sensors = _make_sensors()
        assert self._sim_frequency >= self._robot.frequency >= self._action_frequency

        # self._env.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._terrain = self._terrain_generator(self._env)
        self._env.setGravity(0, 0, -9.8)
        self._robot.update_observation()

        self._num_action_repeats = int(self._sim_frequency / self._action_frequency)
        self._num_execution_repeats = int(self._sim_frequency / self._robot.frequency)
        print(f'Action Repeats for {self._num_action_repeats} time(s)')
        print(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        print(self.observation_dim)
        print(self.observation_space)
        self._sim_step_counter = 0
        # self._num_action_repeats = self._action_time_step / self

    def update_observation(self, observation):
        return self._process_sensors()

    @property
    def robot(self):
        return self._robot

    @property
    def action_space(self):
        return spaces.Box(*np.array(self._robot.action_limits), dtype=np.float64)

    @property
    def observation_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,), dtype=np.float64)

    @property
    def reward_range(self):
        pass

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def step(self, action):
        for _ in range(self._num_action_repeats):
            update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            if update_execution:
                self._robot.apply_command(action)
            self._env.stepSimulation()
            if update_execution:
                self._robot.update_observation()
            self._sim_step_counter += 1

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # for i in np.linspace(0, 2 * np.pi):
    #     print(trajectory_generator(i))

    v = 0.2
    make_robot = make_class(QuadrupedSim, on_rack=True, make_sensors=[OrientationRpySensor, GravityVectorSensor])
    # print(make_robot.__closure__)
    # for c in make_robot.__closure__:
    #     print(c.cell_contents)
    e = LocomotionEnv(make_robot=make_robot)

    # print(e.action_space)
    q = e.robot
    frequency = 240
    gait_frequency = 1
    phase_step = gait_frequency / frequency * 4
    phase = 0
    addition = np.array((0., 0., 0.))

    for _ in range(100000):
        phase_p = phase % 4
        phase_p2 = (phase + 2) % 4
        base_pos = np.array((0, 0, -0.3))
        addition1 = (v / gait_frequency * (2 - abs(phase_p - 2)) / 2, 0, trajectory_generator(phase_p))
        if phase > 2:
            addition2 = (v / gait_frequency * (2 - abs(phase_p2 - 2)) / 2, 0, trajectory_generator(phase_p2))
        else:
            addition2 = (0, 0, 0)
        cmd0 = q.ik(0, base_pos + addition1, 'shoulder')
        cmd1 = q.ik(1, base_pos + addition2, 'shoulder')
        # cmd1 = q.ik(1, base_pos + addition, 'shoulder')
        cmd2 = q.ik(2, base_pos + addition2, 'shoulder')
        cmd3 = q.ik(3, base_pos + addition1, 'shoulder')
        tq = e.step(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
        phase += phase_step
        time.sleep(1. / 240.)
