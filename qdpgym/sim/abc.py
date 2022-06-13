import abc
import dataclasses
import multiprocessing as mp
from typing import Union, Type

import gym
import numpy as np

ARRAY_LIKE = Union[np.ndarray, list, tuple]
NUMERIC = Union[int, float]


class QuadrupedHandle(metaclass=abc.ABCMeta):
    @property
    def obs_history(self):
        raise NotImplementedError

    @property
    def cmd_history(self):
        raise NotImplementedError

    def get_base_pos(self) -> np.ndarray:
        """Get base position in WORLD frame"""
        raise NotImplementedError

    def get_base_orn(self) -> np.ndarray:
        """Get base quaternion in [x, y, z, w]"""
        raise NotImplementedError

    def get_base_rot(self) -> np.ndarray:
        """Get base 3x3 rotation matrix to WORLD frame"""
        raise NotImplementedError

    def get_base_rpy(self) -> np.ndarray:
        """
        Get base Tait–Bryan angles in z-y-x
        @return: np.ndarray: [roll, pitch, yaw]
        """
        raise NotImplementedError

    def get_base_rpy_rate(self) -> np.ndarray:
        """
        Get the rate of base rpy (not angular velocity)
        """
        raise NotImplementedError

    def get_base_lin(self) -> np.ndarray:
        """Get base linear velocity in WORLD frame"""
        raise NotImplementedError

    def get_base_ang(self) -> np.ndarray:
        """Get base angular velocity in WORLD frame"""
        raise NotImplementedError

    def get_velocimeter(self) -> np.ndarray:
        """Get base linear velocity in BASE frame"""
        raise NotImplementedError

    def get_gyro(self) -> np.ndarray:
        """Get base angular velocity in BASE frame"""
        raise NotImplementedError

    def get_accelerometer(self) -> np.ndarray:
        """Get base acceleration in BASE frame"""
        raise NotImplementedError

    def get_state_history(self, latency):
        raise NotImplementedError

    def get_cmd_history(self, latency):
        raise NotImplementedError

    def get_torso_contact(self):
        raise NotImplementedError

    def get_leg_contacts(self):
        raise NotImplementedError

    def get_foot_pos(self):
        raise NotImplementedError

    def get_foot_contacts(self):
        raise NotImplementedError

    def get_contact_forces(self):
        raise NotImplementedError

    def get_force_sensor(self):
        raise NotImplementedError

    def get_slip_vel(self):
        raise NotImplementedError

    def get_strides(self):
        raise NotImplementedError

    def get_clearances(self):
        raise NotImplementedError

    def get_joint_pos(self) -> np.ndarray:
        raise NotImplementedError

    def get_joint_vel(self) -> np.ndarray:
        raise NotImplementedError

    def get_joint_acc(self) -> np.ndarray:
        raise NotImplementedError

    def get_last_command(self) -> np.ndarray:
        raise NotImplementedError

    def get_last_torque(self) -> np.ndarray:
        raise NotImplementedError


class Quadruped(QuadrupedHandle, metaclass=abc.ABCMeta):
    STANCE_HEIGHT: float
    STANCE_CONFIG: tuple

    @property
    def noisy(self) -> QuadrupedHandle:
        return self

    def set_init_pose(self, x=0., y=0., yaw=0.):
        raise NotImplementedError

    def set_random_dynamics(self, flag: bool = True):
        raise NotImplementedError

    def set_latency(self, lower: float, upper=None):
        raise NotImplementedError

    @classmethod
    def inverse_kinematics(cls, leg: int, pos: ARRAY_LIKE):
        raise NotImplementedError

    @classmethod
    def forward_kinematics(cls, leg: int, angles: ARRAY_LIKE):
        raise NotImplementedError


class Terrain(metaclass=abc.ABCMeta):
    def get_height(self, x, y):
        raise NotImplementedError

    def get_normal(self, x, y):
        raise NotImplementedError

    def get_peak(self, x_range, y_range):
        raise NotImplementedError

    def out_of_range(self, x, y) -> bool:
        raise NotImplementedError


@dataclasses.dataclass
class Snapshot(object):
    position: np.ndarray = None
    orientation: np.ndarray = None
    rotation: np.ndarray = None
    rpy: np.ndarray = None
    linear_vel: np.ndarray = None
    angular_vel: np.ndarray = None
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    joint_acc: np.ndarray = None
    foot_pos: np.ndarray = None
    velocimeter: np.ndarray = None
    gyro: np.ndarray = None
    accelerometer: np.ndarray = None
    torso_contact: bool = None
    leg_contacts: np.ndarray = None
    contact_forces: np.ndarray = None
    force_sensor: np.ndarray = None
    rpy_rate: np.ndarray = None


@dataclasses.dataclass
class Command:
    command: np.ndarray = None
    torque: np.ndarray = None


class ComposedObs(tuple):
    pass


class Environment(gym.Env, metaclass=abc.ABCMeta):
    @property
    def robot(self) -> QuadrupedHandle:
        raise NotImplementedError

    @property
    def arena(self) -> Terrain:
        raise NotImplementedError

    @arena.setter
    def arena(self, value):
        raise NotImplementedError

    @property
    def action_history(self):
        raise NotImplementedError

    @property
    def sim_time(self):
        raise NotImplementedError

    @property
    def num_substeps(self):
        raise NotImplementedError

    @property
    def timestep(self):
        raise NotImplementedError

    @property
    def identifier(self):
        raise NotImplementedError

    def get_action_rate(self) -> np.ndarray:
        raise NotImplementedError

    def get_action_accel(self) -> np.ndarray:
        raise NotImplementedError

    def get_relative_robot_height(self) -> float:
        raise NotImplementedError

    def get_interact_terrain_normal(self):
        raise NotImplementedError

    def get_interact_terrain_rot(self) -> np.ndarray:
        raise NotImplementedError

    def get_perturbation(self, in_robot_frame=False):
        raise NotImplementedError

    def set_perturbation(self, value=None):
        raise NotImplementedError


class Hook(metaclass=abc.ABCMeta):
    def register_task(self, task):
        pass

    def initialize(self, robot, env):
        pass

    def init_episode(self, robot, env):
        pass

    def before_step(self, robot, env):
        pass

    def before_substep(self, robot, env):
        pass

    def after_step(self, robot, env):
        pass

    def after_substep(self, robot, env):
        pass

    def on_success(self, robot, env):
        pass

    def on_fail(self, robot, env):
        pass


class CommHook(Hook):
    def __init__(self, comm: mp.Queue):
        self._comm = comm
        self._env_id: str = 'anonymous'

    def initialize(self, robot, env):
        self._env_id = env.identifier

    def _submit(self, info):
        self._comm.put((self._env_id, info))


class CommHookFactory(object):
    def __init__(self, cls: Type[CommHook]):
        self._cls = cls
        self._comm = mp.Queue()

    def __call__(self, *args, **kwargs):
        return self._cls(self._comm, *args, **kwargs)


class Task(metaclass=abc.ABCMeta):
    """
    The task prototype.
    A task is embedded in an environment, whose methods are
    automatically called in different periods of an episode;
    It should also manage hook callbacks.
    Only `before_step` should have return values, which
    should turn action into desired joint angles.
    """

    observation_space: gym.Space
    action_space: gym.Space

    def init_episode(self):
        pass

    def before_step(self, action):
        pass

    def before_substep(self):
        pass

    def after_step(self):
        pass

    def after_substep(self):
        pass

    def on_success(self):
        pass

    def on_fail(self):
        pass

    def register_env(self, robot: Quadruped, env: Environment):
        raise NotImplementedError

    def add_hook(self, hook: Hook, name=None):
        raise NotImplementedError

    def remove_hook(self, name=None):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self, detailed=True):
        """
        Get the reward sum in a step.
        Rewards should be calculated in `after_step` or/and `after_substep`.
        :param detailed: returns an extra dict containing all reward terms.
        :return: reward(, details)
        """
        raise NotImplementedError

    def is_succeeded(self):
        raise NotImplementedError

    def is_failed(self):
        raise NotImplementedError
