import collections
from typing import Union, Any, Optional, Tuple

import gym
import numpy as np
import pybullet as pyb
import pybullet_data as pyb_data
from pybullet_utils.bullet_client import BulletClient

from qdpgym.sim.abc import Task, Environment, ARRAY_LIKE
from qdpgym.sim.blt.quadruped import Aliengo
from qdpgym.sim.blt.terrain import TerrainBase
from qdpgym.utils import PadWrapper, tf


class QuadrupedEnv(Environment):
    _observation_space = None
    _action_space = None

    def __init__(
        self,
        robot: Aliengo,
        arena: TerrainBase,
        task: Task,
        timestep: float = 2e-3,
        time_limit: float = 20,
        num_substeps: int = 10,
        identifier: str = None,
    ):
        self._robot = robot
        self._arena = arena
        self._task = task
        self._timestep = timestep
        self._time_limit = time_limit
        self._num_substeps = num_substeps
        self._identifier = identifier or f'{hex(id(self))[-7:]}'
        self._step_freq = 1 / (self.timestep * self._num_substeps)
        self._render = False

        self._init = False
        self._elapsed_sim_steps = None

        self._sim_env = None if True else pyb  # for pylint
        self._reset_times, self._debug_reset_param = 0, -1
        self._task.register_env(self._robot, self)
        self._action_history = collections.deque(maxlen=10)
        self._perturbation = None

        self._interact_terrain_samples = []
        self._interact_terrain_normal = None
        self._interact_terrain_height = 0.0

    @property
    def observation_space(self) -> gym.Space:
        if self._observation_space is None:
            if hasattr(self._task, 'observation_space'):
                self._observation_space = self._task.observation_space
            else:
                self._observation_space = gym.Space()
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        if self._action_space is None:
            if hasattr(self._task, 'action_space'):
                self._action_space = self._task.action_space
            else:
                self._action_space = gym.Space((12,), float)
        return self._action_space

    def render(self, mode="human"):
        if mode == 'human':
            self._render = True
            if self._sim_env is not None:
                self._sim_env.configureDebugVisualizer(
                    pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True
                )
        else:
            return super().render(mode)

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[Any, Tuple[Any, dict]]:

        super().reset(seed=seed, return_info=return_info, options=options)

        self._init = True
        self._elapsed_sim_steps = 0
        self._action_history.clear()

        if self._sim_env is None:
            if self._render:
                fps = int(self._step_freq)
                self._sim_env = BulletClient(pyb.GUI, options=f'--width=1024 --height=768 --mp4fps={fps}')
                self._debug_reset_param = self._sim_env.addUserDebugParameter('reset', 1, 0, 0)
            else:
                self._sim_env = BulletClient(pyb.DIRECT)
            self._sim_env.setAdditionalSearchPath(pyb_data.getDataPath())
            self._sim_env.setTimeStep(self._timestep)
            self._sim_env.setGravity(0, 0, -9.8)

        self._task.init_episode()
        self._arena.spawn(self._sim_env)
        self._robot.add_to(self._arena)
        self._robot.spawn(self._sim_env, self.np_random)

        for i in range(50):
            self._robot.update_observation(None, minimal=True)
            self._robot.apply_command(self._robot.STANCE_CONFIG)
            self._sim_env.stepSimulation()

        self._action_history.append(np.array(self._robot.STANCE_CONFIG))
        self._robot.update_observation(self.np_random)

        return (
            self._task.get_observation()
            # if return_info else (self._task.get_observation(), {})
        )

    def close(self):
        if self._sim_env is not None:
            self._sim_env.disconnect()

    @property
    def robot(self):
        return self._robot

    @property
    def sim_env(self):
        return self._sim_env

    @property
    def render_mode(self):
        return self._render

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, value: TerrainBase):
        value.replace(self._sim_env, self._arena)
        self._arena = value

    @property
    def action_history(self):
        return PadWrapper(self._action_history)

    @property
    def sim_time(self):
        return self._elapsed_sim_steps * self._timestep

    @property
    def timestep(self):
        return self._timestep

    @property
    def num_substeps(self):
        return self._num_substeps

    @property
    def identifier(self):
        return self._identifier

    def step(self, action: ARRAY_LIKE):
        if self._check_debug_reset_param():
            return self.reset(), 0., True, {}

        assert self._init, 'Call `init_episode` before `step`!'
        action = self._task.before_step(action)
        action = np.asarray(action)
        prev_action = self._action_history[-1]
        self._action_history.append(action)

        for i in range(self._num_substeps):
            weight = (i + 1) / self._num_substeps
            current_action = action * weight + prev_action * (1 - weight)
            self._robot.apply_command(current_action)
            self._task.before_substep()

            self._apply_perturbation()
            self._sim_env.stepSimulation()
            self._elapsed_sim_steps += 1
            self._update_observation()

            self._task.after_substep()
        self._task.after_step()

        done = False
        info = {}
        if self._task.is_failed():
            done = True
            self._task.on_fail()
        elif ((self._time_limit is not None and self.sim_time >= self._time_limit)
              or self._task.is_succeeded()):
            done = True
            info['TimeLimit.truncated'] = True
            self._task.on_success()

        reward, reward_info = self._task.get_reward(True)
        info['reward_info'] = reward_info
        return (
            self._task.get_observation(), reward, done, info
        )

    def _update_observation(self):
        self._robot.update_observation(self.np_random)

        self._interact_terrain_samples.clear()
        self._interact_terrain_height = 0.
        xy_points = self._robot.get_foot_pos()
        for x, y, _ in xy_points:
            h = self._arena.get_height(x, y)
            self._interact_terrain_height += h
            self._interact_terrain_samples.append((x, y, h))
        self._interact_terrain_height /= 4
        self._interact_terrain_normal = tf.estimate_normal(self._interact_terrain_samples)

    def get_action_rate(self) -> np.ndarray:
        if len(self._action_history) < 2:
            return np.zeros(12)
        actions = [self._action_history[-i - 1] for i in range(2)]
        return (actions[0] - actions[1]) * self._step_freq

    def get_action_accel(self) -> np.ndarray:
        if len(self._action_history) < 3:
            return np.zeros(12)
        actions = [self._action_history[-i - 1] for i in range(3)]
        return (actions[0] - 2 * actions[1] + actions[2]) * self._step_freq ** 2

    def get_relative_robot_height(self) -> float:
        return self._robot.get_base_pos()[2] - self._interact_terrain_height

    def get_interact_terrain_normal(self):
        return self._interact_terrain_normal

    def get_interact_terrain_rot(self) -> np.ndarray:
        return tf.Rotation.from_zaxis(self._interact_terrain_normal)

    def get_perturbation(self, in_robot_frame=False):
        if self._perturbation is None:
            return None
        elif in_robot_frame:
            rotation = self._robot.get_base_rot()
            perturbation = np.concatenate(
                [rotation.T @ self._perturbation[i * 3:i * 3 + 3] for i in range(2)]
            )
        else:
            perturbation = self._perturbation
        return perturbation

    def set_perturbation(self, value=None):
        if value is None:
            self._perturbation = None
        else:
            self._perturbation = np.array(value)

    def _apply_perturbation(self):
        if self._perturbation is not None:
            self._applied_link_id = 0
            self._sim_env.applyExternalForce(
                objectUniqueId=self._robot.id,
                linkIndex=self._applied_link_id,
                forceObj=self._perturbation[:3],
                posObj=self._robot.get_base_pos(),
                flags=pyb.WORLD_FRAME
            )
            self._sim_env.applyExternalTorque(
                objectUniqueId=self._robot.id,
                linkIndex=self._applied_link_id,
                torqueObj=self._perturbation[3:],
                flags=pyb.WORLD_FRAME
            )

    def _check_debug_reset_param(self):
        if self._debug_reset_param != -1:
            reset_times = self._sim_env.readUserDebugParameter(self._debug_reset_param)
            if reset_times != self._reset_times:
                self._reset_times = reset_times
                return True
        return False
