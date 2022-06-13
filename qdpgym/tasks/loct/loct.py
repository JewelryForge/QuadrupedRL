import collections
import math
import queue
import sys
import time
from typing import Optional

import gym.spaces
import numpy as np

import qdpgym.tasks.loct.reward as all_rewards
# import qdpgym.tasks.loct.sr_reward as all_rewards
from qdpgym.sim.abc import Quadruped, Environment, QuadrupedHandle, Hook, CommHook, CommHookFactory
from qdpgym.sim.common.tg import TgStateMachine, vertical_tg
from qdpgym.sim.task import BasicTask
from qdpgym.utils import tf, log, PadWrapper


class LocomotionV0(BasicTask):
    ALL_REWARDS = all_rewards

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(211,))
    action_space = gym.spaces.Box(low=-1., high=1., shape=(12,))

    def __init__(self, substep_reward_on=True):
        super().__init__(substep_reward_on)
        self._cmd = np.array((0., 0., 0.))
        self._traj_gen: Optional[TgStateMachine] = None

        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._action_weights = np.array((0.3, 0.2, 0.1) * 4)

        self._target_history = collections.deque(maxlen=10)

    @property
    def cmd(self):
        return self._cmd.copy()

    @cmd.setter
    def cmd(self, value):
        self._cmd = np.array(value)

    @property
    def np_random(self):
        return self._env.np_random

    @property
    def target_history(self):
        return PadWrapper(self._target_history)

    def register_env(self, robot, env):
        super().register_env(robot, env)
        robot.set_latency(0., 0.03)
        robot.set_random_dynamics(True)
        self._traj_gen = TgStateMachine(
            env.timestep * env.num_substeps,
            self.np_random,
            vertical_tg(0.12)
        )
        self._build_weights_and_bias()

    def init_episode(self):
        self._robot.set_init_pose(
            yaw=self._env.np_random.random() * 2 * np.pi
        )
        self._traj_gen.reset()
        super().init_episode()

    def before_step(self, action):
        super().before_step(action)
        action = action * self._action_weights
        self._traj_gen.update()
        priori = self._traj_gen.get_priori_trajectory().reshape(4, 3)
        des_pos = action.reshape(4, 3) + priori
        self._target_history.append(des_pos)
        des_joint_pos = []
        for i, pos in enumerate(des_pos):
            des_joint_pos.append(self._robot.inverse_kinematics(i, pos))
        return np.concatenate(des_joint_pos)

    def get_observation(self):
        r: Quadruped = self._robot
        e: Environment = self._env
        n: QuadrupedHandle = r.noisy
        if (self._cmd[:2] == 0.).all():
            cmd_obs = np.concatenate(((0.,), self._cmd))
        else:
            cmd_obs = np.concatenate(((1.,), self._cmd))

        roll_pitch = n.get_base_rpy()[:2]
        base_linear = n.get_velocimeter()
        base_angular = n.get_gyro()
        joint_pos = n.get_joint_pos()
        joint_vel = n.get_joint_vel()

        action_history = e.action_history
        joint_target = action_history[-1]
        joint_target_history = action_history[-2]

        tg_base_freq = (self._traj_gen.base_frequency,)
        tg_freq = self._traj_gen.frequency
        tg_raw_phases = self._traj_gen.phases
        tg_phases = np.concatenate((np.sin(tg_raw_phases), np.cos(tg_raw_phases)))

        joint_err = n.get_last_command() - n.get_joint_pos()
        state1, state2 = n.get_state_history(0.01), n.get_state_history(0.02)
        cmd1, cmd2 = n.get_cmd_history(0.01).command, n.get_cmd_history(0.02).command
        joint_proc_err = np.concatenate((cmd1 - state1.joint_pos, cmd2 - state2.joint_pos))
        joint_proc_vel = np.concatenate((state1.joint_vel, state2.joint_vel))

        terrain_info = self._collect_terrain_info()
        contact_states = r.get_leg_contacts()
        contact_forces = r.get_force_sensor().reshape(-1)
        foot_friction_coeffs = np.ones(4)

        perturbation = e.get_perturbation(in_robot_frame=True)
        if perturbation is None:
            perturbation = np.zeros(6)

        return (np.concatenate((
            terrain_info,
            contact_states,
            contact_forces,
            foot_friction_coeffs,
            perturbation,
            cmd_obs,
            roll_pitch,
            base_linear,
            base_angular,
            joint_pos,
            joint_vel,
            joint_target,
            tg_phases,
            tg_freq,
            joint_err,
            joint_proc_err,
            joint_proc_vel,
            joint_target_history,
            tg_base_freq
        )) - self._bias) * self._weights

    def is_succeeded(self):
        x, y, _ = self._robot.get_base_pos()
        return self._env.arena.out_of_range(x, y)

    def is_failed(self):
        r = self._robot.get_base_rpy()[0]
        x, y, _ = self._robot.get_base_pos()
        rel_h = self._env.get_relative_robot_height()
        if (
            rel_h < self._robot.STANCE_HEIGHT * 0.5 or
            rel_h > self._robot.STANCE_HEIGHT * 1.5 or
            r < -np.pi / 3 or r > np.pi / 3 or
            self._robot.get_torso_contact()
        ):
            return True
        return False

    def _collect_terrain_info(self):
        yaw = self._robot.get_base_rpy()[2]
        dx, dy = 0.1 * math.cos(yaw), 0.1 * math.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        scan = []
        for x, y, z in self._robot.get_foot_pos():
            for px, py in points:
                scan.append(z - self._env.arena.get_height(x + px, y + py))

        slopes = []
        for x, y, z in self._robot.get_foot_pos():
            trnZ = self._env.arena.get_normal(x, y)
            sy, cy = np.sin(yaw), np.cos(yaw)
            trnX = tf.vcross((-sy, cy, 0), trnZ)
            trnX /= tf.vnorm(trnX)
            trnY = tf.vcross(trnZ, trnX)
            slopes.extend((np.arcsin(trnX[2]), np.arcsin(trnY[2])))
        return np.concatenate((scan, slopes))

    def _build_weights_and_bias(self):
        self._weights = np.concatenate((
            (5.,) * 36,  # terrain scan
            (2.5,) * 8,  # terrain slope
            (2.,) * 12,  # contact
            (0.01, 0.01, 0.02) * 4,  # contact force
            (1.,) * 4,  # friction,
            (0.1, 0.1, 0.1, 0.4, 0.2, 0.2),  # perturbation
            (1.,) * 4,  # command
            (2., 2.),  # roll pitch
            (2.,) * 3,  # linear
            (2.,) * 3,  # angular
            (2.,) * 12,  # joint pos
            (0.5, 0.4, 0.3) * 4,  # joint vel
            (2.,) * 12,  # joint target
            (1.,) * 8,  # tg phases
            (100.,) * 4,  # tg freq
            (6.5, 4.5, 3.5) * 4,  # joint error
            (5.,) * 24,  # proc joint error
            (0.5, 0.4, 0.3) * 8,  # proc joint vel
            (2.,) * 12,  # joint target history
            (1,)  # tg base freq
        ))
        stance_cfg = self._robot.STANCE_CONFIG
        self._bias = np.concatenate((
            (0.,) * 36,  # terrain scan
            (0,) * 8,  # terrain slope
            (0.5,) * 12,  # contact
            (0., 0., 30.) * 4,  # contact force
            (0.,) * 4,  # friction,
            (0.,) * 6,  # perturbation
            (0.,) * 4,  # command
            (0., 0.),  # roll pitch
            (0.,) * 3,  # linear
            (0.,) * 3,  # angular
            stance_cfg,  # joint pos
            (0.,) * 12,  # joint vel
            stance_cfg,  # joint target
            (0.,) * 8,  # tg phases
            (self._traj_gen.base_frequency,) * 4,  # tg freq
            (0.,) * 12,  # joint error
            (0.,) * 24,  # joint proc error
            (0.,) * 24,  # joint proc vel
            stance_cfg,  # joint target history
            (self._traj_gen.base_frequency,)  # tg base freq
        ))


class LocomotionSimple(LocomotionV0):
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(133,))

    def get_observation(self):
        r: Quadruped = self._robot
        e: Environment = self._env
        n: QuadrupedHandle = r.noisy
        if (self._cmd[:2] == 0.).all():
            cmd_obs = np.concatenate(((0.,), self._cmd))
        else:
            cmd_obs = np.concatenate(((1.,), self._cmd))

        roll_pitch = n.get_base_rpy()[:2]
        base_linear = n.get_velocimeter()
        base_angular = n.get_gyro()
        joint_pos = n.get_joint_pos()
        joint_vel = n.get_joint_vel()

        action_history = e.action_history
        joint_target = action_history[-1]
        joint_target_history = action_history[-2]

        tg_base_freq = (self._traj_gen.base_frequency,)
        tg_freq = self._traj_gen.frequency
        tg_raw_phases = self._traj_gen.phases
        tg_phases = np.concatenate((np.sin(tg_raw_phases), np.cos(tg_raw_phases)))

        joint_err = n.get_last_command() - n.get_joint_pos()
        state1, state2 = n.get_state_history(0.01), n.get_state_history(0.02)
        cmd1, cmd2 = n.get_cmd_history(0.01).command, n.get_cmd_history(0.02).command
        joint_proc_err = np.concatenate((cmd1 - state1.joint_pos, cmd2 - state2.joint_pos))
        joint_proc_vel = np.concatenate((state1.joint_vel, state2.joint_vel))

        return (np.concatenate((
            cmd_obs,
            roll_pitch,
            base_linear,
            base_angular,
            joint_pos,
            joint_vel,
            joint_target,
            tg_phases,
            tg_freq,
            joint_err,
            joint_proc_err,
            joint_proc_vel,
            joint_target_history,
            tg_base_freq
        )) - self._bias) * self._weights

    def _build_weights_and_bias(self):
        self._weights = np.concatenate((
            (1.,) * 4,  # command
            (2., 2.),  # roll pitch
            (2.,) * 3,  # linear
            (2.,) * 3,  # angular
            (2.,) * 12,  # joint pos
            (0.5, 0.4, 0.3) * 4,  # joint vel
            (2.,) * 12,  # joint target
            (1.,) * 8,  # tg phases
            (100.,) * 4,  # tg freq
            (6.5, 4.5, 3.5) * 4,  # joint error
            (5.,) * 24,  # proc joint error
            (0.5, 0.4, 0.3) * 8,  # proc joint vel
            (2.,) * 12,  # joint target history
            (1,)  # tg base freq
        ))
        stance_cfg = self._robot.STANCE_CONFIG
        self._bias = np.concatenate((
            (0.,) * 4,  # command
            (0., 0.),  # roll pitch
            (0.,) * 3,  # linear
            (0.,) * 3,  # angular
            stance_cfg,  # joint pos
            (0.,) * 12,  # joint vel
            stance_cfg,  # joint target
            (0.,) * 8,  # tg phases
            (self._traj_gen.base_frequency,) * 4,  # tg freq
            (0.,) * 12,  # joint error
            (0.,) * 24,  # joint proc error
            (0.,) * 24,  # joint proc vel
            stance_cfg,  # joint target history
            (self._traj_gen.base_frequency,)  # tg base freq
        ))


class LocomotionPMTG(LocomotionV0):
    action_space = gym.spaces.Box(low=-1., high=1., shape=(16,))

    def __init__(self):
        super().__init__()
        self._freq_weights = np.array((0.5,) * 4)

    def register_env(self, robot, env):
        super().register_env(robot, env)
        robot.set_latency(0., 0.03)
        robot.set_random_dynamics(True)
        self._traj_gen = TgStateMachine(
            env.timestep * env.num_substeps,
            self.np_random,
            vertical_tg(0.12),
            # 'random'
        )
        self._build_weights_and_bias()

    def before_step(self, action):
        super(LocomotionV0, self).before_step(action)
        foot_pos_res = action[4:] * self._action_weights
        freq_offset = action[:4] * self._freq_weights
        self._traj_gen.update(freq_offset)
        priori = self._traj_gen.get_priori_trajectory().reshape(4, 3)
        des_foot_pos = foot_pos_res.reshape(4, 3) + priori
        self._target_history.append(des_foot_pos)
        des_joint_pos = []
        for i, pos in enumerate(des_foot_pos):
            des_joint_pos.append(self._robot.inverse_kinematics(i, pos))
        return np.concatenate(des_joint_pos)


# class LocomotionV0Raw(LocomotionV0):
#     def get_observation(self):
#         r: Quadruped = self._robot
#         e: Environment = self._env
#         n: QuadrupedHandle = r.noisy
#         if (self._cmd[:2] == 0.).all():
#             cmd_obs = np.concatenate(((0.,), self._cmd))
#         else:
#             cmd_obs = np.concatenate(((1.,), self._cmd))
#
#         n_roll_pitch = n.get_base_rpy()[:2]
#         n_base_linear = n.get_velocimeter()
#         n_base_angular = n.get_gyro()
#         n_joint_pos = n.get_joint_pos()
#         n_joint_vel = n.get_joint_vel()
#
#         r_roll_pitch = r.get_base_rpy()[:2]
#         r_base_linear = r.get_velocimeter()
#         r_base_angular = r.get_gyro()
#         r_joint_pos = r.get_joint_pos()
#         r_joint_vel = r.get_joint_vel()
#
#         action_history = e.action_history
#         joint_target = action_history[-1]
#         joint_target_history = action_history[-2]
#
#         tg_base_freq = (self._traj_gen.base_frequency,)
#         tg_freq = self._traj_gen.frequency
#         tg_raw_phases = self._traj_gen.phases
#         tg_phases = np.concatenate((np.sin(tg_raw_phases), np.cos(tg_raw_phases)))
#
#         n_joint_err = n.get_last_command() - n.get_joint_pos()
#         n_state1, n_state2 = n.get_state_history(0.01), n.get_state_history(0.02)
#         cmd1, cmd2 = n.get_cmd_history(0.01).command, n.get_cmd_history(0.02).command
#         n_joint_proc_err = np.concatenate((cmd1 - n_state1.joint_pos, cmd2 - n_state2.joint_pos))
#         n_joint_proc_vel = np.concatenate((n_state1.joint_vel, n_state2.joint_vel))
#
#         r_joint_err = r.get_last_command() - r.get_joint_pos()
#         r_state1, r_state2 = r.get_state_history(0.01), r.get_state_history(0.02)
#         r_joint_proc_err = np.concatenate((cmd1 - r_state1.joint_pos, cmd2 - r_state2.joint_pos))
#         r_joint_proc_vel = np.concatenate((r_state1.joint_vel, r_state2.joint_vel))
#
#         terrain_info = self._collect_terrain_info()
#         contact_states = r.get_leg_contacts()
#         contact_forces = r.get_force_sensor().reshape(-1)
#         foot_friction_coeffs = np.ones(4)
#
#         perturbation = e.get_perturbation(in_robot_frame=True)
#         if perturbation is None:
#             perturbation = np.zeros(6)
#
#         return ComposedObs((
#             (np.concatenate((
#                 terrain_info,
#                 contact_states,
#                 contact_forces,
#                 foot_friction_coeffs,
#                 perturbation,
#                 cmd_obs,
#                 n_roll_pitch,
#                 n_base_linear,
#                 n_base_angular,
#                 n_joint_pos,
#                 n_joint_vel,
#                 joint_target,
#                 tg_phases,
#                 tg_freq,
#                 n_joint_err,
#                 n_joint_proc_err,
#                 n_joint_proc_vel,
#                 joint_target_history,
#                 tg_base_freq
#             )) - self._bias) * self._weights,
#             (np.concatenate((
#                 terrain_info,
#                 contact_states,
#                 contact_forces,
#                 foot_friction_coeffs,
#                 perturbation,
#                 cmd_obs,
#                 r_roll_pitch,
#                 r_base_linear,
#                 r_base_angular,
#                 r_joint_pos,
#                 r_joint_vel,
#                 joint_target,
#                 tg_phases,
#                 tg_freq,
#                 r_joint_err,
#                 r_joint_proc_err,
#                 r_joint_proc_vel,
#                 joint_target_history,
#                 tg_base_freq
#             )) - self._bias) * self._weights
#         ))


class RandomCommanderHookV0(Hook):
    def __init__(self):
        self._task: Optional[LocomotionV0] = None
        self._stop_prob = 0.2
        self._interval_range = (0.5, 5.)
        self._interval = 0
        self._last_update = 0

    def register_task(self, task):
        self._task = task

    def get_random_cmd(self, random_state):
        angular_cmd = random_state.choice((-1., 0, 0, 1.))
        cmd_choice = random_state.random()
        if cmd_choice < self._stop_prob:
            return np.array((0., 0., angular_cmd))
        else:
            yaw = random_state.uniform(0, math.tau)
            return np.array((math.cos(yaw), math.sin(yaw), angular_cmd))

    def init_episode(self, robot, env):
        random = env.np_random
        self._task.cmd = self.get_random_cmd(random)
        self._interval = random.uniform(*self._interval_range)

    def after_step(self, robot, env):
        if env.sim_time >= self._last_update + self._interval:
            random = env.np_random
            self._task.cmd = self.get_random_cmd(random)
            self._last_update = env.sim_time
            self._interval = random.uniform(*self._interval_range)


class RandomCommanderHookV1(RandomCommanderHookV0):
    def get_random_cmd(self, random_state):
        angular_cmd = random_state.uniform(-1., 1.)
        cmd_choice = random_state.random()
        if cmd_choice < self._stop_prob:
            return np.array((0., 0., angular_cmd))
        else:
            yaw = random_state.uniform(0, 2 * np.pi)
            mag = random_state.random()
            return np.array((
                math.cos(yaw) * mag,
                math.sin(yaw) * mag,
                angular_cmd
            ))


class CommandRewardCollectorHook(CommHook):
    def __init__(self, comm, reward_name):
        self._reward_name = reward_name
        super().__init__(comm)
        self._task = None

        self._cmd_buffer = None
        self._reward_sum = 0.
        self._counter = 0

    def register_task(self, task):
        self._task = task

    def after_step(self, robot, env):
        if np.array_equal(self._cmd_buffer, self._task.cmd):
            self._counter += 1
            _, reward_info = self._task.get_reward(detailed=True)
            self._reward_sum += reward_info[self._reward_name]
        else:
            if self._cmd_buffer is not None and self._counter:
                self._submit({
                    'command': self._cmd_buffer,
                    'reward': self._reward_sum / self._counter
                })
            self._cmd_buffer = self._task.cmd
            self._reward_sum = 0.
            self._counter = 0


class CommandRewardAnalyser(CommHookFactory):
    def __init__(self):
        super().__init__(CommandRewardCollectorHook)
        self._history = collections.deque(maxlen=500)

    def analyse(self) -> np.ndarray:
        try:
            while True:
                self._history.append(self._comm.get(block=False))
        except queue.Empty:
            pass
        fig = np.zeros((200, 200))
        for env_id, info in self._history:
            x, y, _ = info['command']
            coord_x = 100 + int(-y * 75)
            coord_y = 100 + int(-x * 75)
            fig[coord_x, coord_y] = np.clip(info['reward'], 0.2, 1.)
        return fig


class GamepadCommanderHook(Hook):
    def __init__(self, gamepad_type='Xbox'):
        from qdpgym.thirdparty.gamepad import gamepad, controllers
        if not gamepad.available():
            log.warn('Please connect your gamepad...')
            while not gamepad.available():
                time.sleep(1.0)
        try:
            self.gamepad: gamepad.Gamepad = getattr(controllers, gamepad_type)()
            self.gamepad_type = gamepad_type
        except AttributeError:
            raise RuntimeError(f'`{gamepad_type}` is not supported,'
                               f'all {controllers.all_controllers}')
        self.gamepad.startBackgroundUpdates()
        log.info('Gamepad connected')

        self._task: Optional[LocomotionV0] = None

    @classmethod
    def is_available(cls):
        from qdpgym.thirdparty.gamepad import gamepad
        return gamepad.available()

    def register_task(self, task):
        self._task = task

    def before_step(self, robot, env):
        if self.gamepad.isConnected():
            x_speed = -self.gamepad.axis('LAS -Y')
            y_speed = -self.gamepad.axis('LAS -X')
            steering = -self.gamepad.axis('RAS -X')
            self._task.cmd = (x_speed, y_speed, steering)
        else:
            sys.exit(1)

    def __del__(self):
        self.gamepad.disconnect()
