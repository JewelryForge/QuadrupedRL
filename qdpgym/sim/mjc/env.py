import collections
from typing import Optional

import numpy as np
from dm_control import mjcf
from dm_control.composer import Arena

from qdpgym.sim.abc import Task, Environment, ARRAY_LIKE, TimeStep, StepType
from .quadruped import Aliengo
from qdpgym.utils import PadWrapper


class QuadrupedEnv(Environment):
    def __init__(self, robot: Aliengo, arena: Arena, task: Task,
                 timestep=2e-3, time_limit: float = None,
                 num_substeps=10, seed=None):
        self._robot = robot
        self._arena = arena
        self._task = task
        self._timestep = timestep
        self._time_limit = time_limit
        self._num_substeps = num_substeps

        self._init = False
        self._num_sim_steps = 0
        self._random = np.random.RandomState(seed)

        self._arena.mjcf_model.option.timestep = timestep
        self._physics: Optional[mjcf.Physics] = None
        self._task.register_env(self._robot, self, self._random)
        self._action_history = collections.deque(maxlen=10)
        self._perturbation = None

    def init_episode(self):
        self._init = True
        self._num_sim_steps = 0
        self._action_history.clear()

        self._robot.add_to(self._arena)
        self._robot.init_mjcf_model(self._random)
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self._task.init_episode()
        self._robot.init_physics(self._physics, self._random)

        for i in range(50):
            self._robot.update_observation(None, True)
            self._robot.apply_command(self._robot.STANCE_CONFIG)
            self._physics.step()

        self._action_history.append(np.array(self._robot.STANCE_CONFIG))
        self._physics.data.ptr.time = 0.
        self._robot.update_observation(self._random)
        return TimeStep(
            StepType.INIT,
            self._task.get_observation()
        )

    @property
    def robot(self):
        return self._robot

    @property
    def physics(self):
        return self._physics

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, value):
        self._init = False
        self._arena = value
        self._robot.add_to(self._arena)

    @property
    def action_history(self):
        return PadWrapper(self._action_history)

    @property
    def sim_time(self):
        return self._physics.time()

    @property
    def timestep(self):
        return self._timestep

    @property
    def num_substeps(self):
        return self._num_substeps

    def step(self, action: ARRAY_LIKE):
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
            self._physics.step()
            self._num_sim_steps += 1
            self._update_observation()
            self._robot.update_observation(self._random)

            self._task.after_substep()
        self._task.after_step()
        if self._task.is_failed():
            status = StepType.FAIL
            self._task.on_fail()
        elif ((self._time_limit is not None and self.sim_time >= self._time_limit)
              or self._task.is_succeeded()):
            status = StepType.SUCCESS
            self._task.on_success()
        else:
            status = StepType.REGULAR
        reward, reward_info = self._task.get_reward(True)
        return TimeStep(
            status,
            self._task.get_observation(),
            reward,
            reward_info
        )

    def _update_observation(self):
        pass

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
        self._perturbation = np.array(value)

    def _apply_perturbation(self):
        if self._perturbation is not None:
            self._physics.bind(self._robot.entity.root_body).xfrc_applied = self._perturbation
