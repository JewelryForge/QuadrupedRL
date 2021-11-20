import time

import gym
import numpy as np
import pybullet
import torch
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client

# from burl.utils.bc import Observable
from burl.rl.state import ObservationRaw, ProprioceptiveObservation, Observation, ExtendedObservation
from burl.sim.config import RenderParam
from burl.sim.quadruped import QuadrupedSim
from burl.sim.sensors import OrientationRpySensor, GravityVectorSensor
from burl.sim.terrain import make_plane
from burl.rl.tg import end_trajectory_generator, LocomotionStateMachine
from burl.utils import make_cls
from burl.rl.task import SimpleForwardTaskOnFlat

# from reward import *
from burl.utils.transforms import Rotation


class LocomotionEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self._render_cfg = kwargs.get('render_config', RenderParam())
        # self._gui = self._render_cfg.rendering_enabled
        self._gui = kwargs.get('use_gui', False)
        if self._gui:
            self._env = bullet_client.BulletClient(pybullet.GUI)
            self._init_rendering()
        else:
            self._env = bullet_client.BulletClient(pybullet.DIRECT)
        if False:
            self._env = pybullet
        # _make_sensors = kwargs.get('make_sensors', [])
        # super().__init__(make_sensors=_make_sensors)
        _make_robot = kwargs.get('make_robot', QuadrupedSim)
        self._robot: QuadrupedSim = _make_robot(sim_env=self._env)
        # self._subordinates.append(self._robot)

        _make_task = kwargs.get('make_task', SimpleForwardTaskOnFlat)
        self._task = _make_task(self)
        self._terrain_generator = kwargs.get('terrain_generator', make_plane)  # TODO: RANDOMLY CHANGE TERRAIN
        self._random_state, self._random_seed = self.seed(kwargs.get('seed', None))
        self._sim_frequency = kwargs.get('sim_frequency', 240)
        self._action_frequency = kwargs.get('action_frequency', 120)
        assert self._sim_frequency >= self._robot.frequency >= self._action_frequency

        self._terrain = self._terrain_generator(self._env)
        self._set_physics_parameters()
        # ONLY USE FOR INIT
        # self._init_robot_observation = self._robot.update_observation()

        self._num_action_repeats = int(self._sim_frequency / self._action_frequency)
        self._num_execution_repeats = int(self._sim_frequency / self._robot.frequency)
        print(f'Action Repeats for {self._num_action_repeats} time(s)')
        print(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        self._sim_step_counter = 0

    def init_observation(self):
        observations = self._robot.update_observation()
        return observations

    # def _on_update_observation(self):
    #     pass

    @property
    def robot(self):
        return self._robot

    @property
    def action_space(self):
        return spaces.Box(*np.array(self._robot.action_limits), dtype=np.float64)

    # @property
    # def observation_space(self):
    #     return spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,), dtype=np.float64)

    # @property
    # def reward_range(self):
    #     pass

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def step(self, action):
        # NOTICE: ADDING LATENCY ARBITRARILY FROM A DISTRIBUTION IS NOT REASONABLE
        # NOTICE: SHOULD CALCULATE TIME_SPENT IN REAL WORLD; HERE USE FIXED TIME INTERVAL
        for _ in range(self._num_action_repeats):
            update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            if update_execution:
                torques = self._robot.apply_command(action)
            self._env.stepSimulation()
            if update_execution:
                privileged_observation, observation = self._robot.update_observation()

            self._sim_step_counter += 1
        if self._gui:
            self._check_render_options()
        info = privileged_observation.__dict__.copy()
        info.update({'action': action, 'torques': torques})
        # print(torques)
        return (privileged_observation, observation), self._task(info), not self._robot.is_safe(), info

    def reset(self, **kwargs):
        completely_reset = kwargs.get('completely_reset', False)
        if completely_reset:
            raise NotImplementedError
        return self._robot.reset()

    def render(self, mode="rgb_array"):
        if mode == 'rgb_array':
            view_matrix = self._env.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._robot.position,
                distance=self._render_cfg.camera_distance,
                roll=0, pitch=self._render_cfg.camera_pitch, yaw=self._render_cfg.camera_yaw,
                upAxisIndex=2)
            proj_matrix = self._env.computeProjectionMatrixFOV(
                fov=60,
                aspect=self._render_cfg.render_width / self._render_cfg.render_height,
                nearVal=0.1,
                farVal=100.0)
            _, _, px, _, _ = self._env.getCameraImage(
                width=self._render_cfg.render_width,
                height=self._render_cfg.render_height,
                renderer=self._env.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        else:
            super().render(mode=mode)

    def close(self):
        self._env.disconnect()

    def _init_rendering(self):
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, True)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)
        # if hasattr(self._task, '_draw_ref_model_alpha'):
        #     self._show_reference_id = pybullet.addUserDebugParameter("show reference", 0, 1,
        #                                                              self._task._draw_ref_model_alpha)
        # self._delay_id = self._env.addUserDebugParameter("delay", 0, 0.3, 0)
        self._dbg_reset = self._env.addUserDebugParameter('reset', 1, 0, 0)
        self._reset_counter = 0

        if self._render_cfg.egl_rendering:  # TODO: WHAT DOES THE PLUGIN DO?
            self._env.loadPlugin('eglRendererPlugin')
        self._last_frame_time = time.time()

    def _set_physics_parameters(self):
        # self._env.resetSimulation()
        # self._env.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
        self._env.setTimeStep(1 / self._sim_frequency)
        self._env.setGravity(0, 0, -9.8)

    def _check_render_options(self):
        if (current := self._env.readUserDebugParameter(self._dbg_reset)) != self._reset_counter:
            self._reset_counter = current
            self.reset()
        if self._render_cfg.sleeping_enabled:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._num_action_repeats / self._sim_frequency - time_spent
            # print(time_spent)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        # Keep the previous orientation of the camera set by the user.
        yaw, pitch, dist = self._env.getDebugVisualizerCamera()[8:11]
        self._env.resetDebugVisualizerCamera(dist, yaw, pitch, self._robot.position)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        # alpha = 1.
        # if self._show_reference_id > 0:
        #     alpha = self._pybullet_client.readUserDebugParameter(self._show_reference_id)

        # ref_col = [1, 1, 1, alpha]
        # if hasattr(self._task, '_ref_model'):
        #     self._pybullet_client.changeVisualShape(self._task._ref_model, -1, rgbaColor=ref_col)
        #     for l in range(self._pybullet_client.getNumJoints(self._task._ref_model)):
        #         self._pybullet_client.changeVisualShape(self._task._ref_model, l, rgbaColor=ref_col)
        #
        # delay = self._pybullet_client.readUserDebugParameter(self._delay_id)
        # if (delay > 0):
        #     time.sleep(delay)


class TGEnv(LocomotionEnv):
    def __init__(self, **kwargs):
        super(TGEnv, self).__init__(**kwargs)
        self._stm = LocomotionStateMachine(1 / self._action_frequency)

    def make_standard_observation(self, observation: ObservationRaw):
        eo = ExtendedObservation()
        eo.command = self._task.cmd
        eo.gravity_vector = Rotation.from_quaternion(observation.base_state.pose.orientation).Z
        eo.base_linear = observation.base_state.twist.linear
        eo.base_angular = observation.base_state.twist.angular
        eo.joint_pos = observation.joint_states.position
        eo.joint_vel = observation.joint_states.velocity
        eo.joint_prev_pos_err = self._robot.get_joint_position_error_history(-1 / self._robot.frequency)
        eo.ftg_frequencies = self._stm.frequency
        eo.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        eo.base_frequency = (self._stm.base_frequency,)
        eo.joint_pos_err_his = np.concatenate((self._robot.get_joint_position_error_history(-0.01),
                                               self._robot.get_joint_position_error_history(-0.02)))
        eo.joint_vel_his = np.concatenate((self._robot.get_joint_velocity_history(-0.01),
                                           self._robot.get_joint_velocity_history(-0.02)))
        eo.joint_pos_target = self._robot.command_history[-1]
        eo.joint_prev_pos_err = self._robot.command_history[-2]

        eo.terrain_scan = np.zeros(36)
        eo.terrain_normal = np.array([0, 0, 1] * 4)
        eo.contact_states = observation.contact_states[1:]
        eo.foot_contact_forces = observation.joint_states.reaction_force[:, 3].reshape(-1)
        eo.foot_friction_coeffs = np.zeros(4)
        eo.external_disturbance = np.zeros(3)
        return eo.to_array()

    def init_observation(self):
        return [self.make_standard_observation(ob) for ob in super().init_observation()]

    def step(self, action):
        # 0 ~ 3 additional frequencies
        # 4 ~ 11 foot position residual
        self._stm.update(action[:4])
        priori = self._stm.get_priori_trajectory()
        residuals = action[4:]
        for i in range(4):
            residuals[i * 3 + 2] += priori[i]
        # print(residuals)
        # NOTICE: HIP IS USED IN PAPER
        commands = [self._robot.ik(i, residuals[i * 3: i * 3 + 3], 'shoulder') for i in range(4)]
        # print(commands)
        observations, reward, done, info = super().step(np.concatenate(commands))
        # TODO: COMPLETE NOISY OBSERVATION CONVERSIONS
        return (self.make_standard_observation(observations[0]),
                reward, done, info)


class EnvContainer(object):
    num_obs = ExtendedObservation.dim
    num_privileged_obs = ExtendedObservation.dim

    def __init__(self, num_envs, make_env, device='cuda'):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = 1000
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float)
        self.extras = {}
        self._envs = [make_env() for _ in range(num_envs)]
        # self._envs = [make_env(use_gui=True)] + [make_env(use_gui=False) for _ in range(num_envs-1)]

    def step(self, actions: torch.Tensor):
        actions = actions.cpu().numpy()
        observations, rewards, dones, _ = zip(*[e.step(a) for e, a in zip(self._envs, actions)])
        self.reset(i for i in range(self.num_envs) if dones[i] is True)
        # print(len(observations))
        return (torch.Tensor(np.array(observations)), None, torch.Tensor(np.array(rewards)),
                torch.Tensor(np.array(dones)), {})

    def reset(self, env_ids):
        for i in env_ids:
            # print(i, 'reset')
            self._envs[i].reset()

    def init_observations(self):
        # print(len(tuple(zip(*(env.init_observation() for env in self._envs)))))
        # print(len(tuple(env.init_observation() for env in self._envs)))
        # print('o', *(o.shape for o in zip(env.init_observation() for env in self._envs)), sep='\n', end='\n\n')
        # TO MY ASTONISHMENT, A LIST COMPREHENSION IS FASTER THAN A GENERATOR!!!
        return (torch.Tensor(np.asarray(o)) for o in zip(*[env.init_observation() for env in self._envs]))


if __name__ == '__main__':
    make_robot = make_cls(QuadrupedSim, on_rack=False, make_sensors=[],
                          frequency=400)
    make_env = make_cls(TGEnv, make_robot=make_robot, sim_frequency=400,
                        action_frequency=50, use_gui=False)
    envs = EnvContainer(4, make_env)
    torch.set_printoptions(2, linewidth=1000)
    envs.init_observations()
    print(*envs.step(torch.Tensor([[0] * 16] * 4)), sep='\n')

    # print(*envs.init_observations(), sep='\n')
# if __name__ == '__main__':
#     # for i in np.linspace(0, 2 * np.pi):
#     #     print(trajectory_generator(i))
#
#     v = 0.3
#     make_robot = make_cls(QuadrupedSim, on_rack=False, make_sensors=[], frequency=400)
#     e = LocomotionEnv(make_robot=make_robot, sim_frequency=400, action_frequency=50, use_gui=True)
#     e.init_observation()
#
#     # print(e.action_space)
#     q = e.robot
#     frequency = 240
#     gait_frequency = 1
#     phase_step = gait_frequency / frequency * 4
#     phase = 0
#     addition = np.array((0., 0., 0.))
#
#     for _ in range(1000):
#         phase_p = phase % 4
#         phase_p2 = (phase + 2) % 4
#         base_pos = np.array((0, 0, -0.3))
#         addition1 = (v / gait_frequency * (2 - abs(phase_p - 2)) / 2, 0, end_trajectory_generator(phase_p))
#         if phase > 2:
#             addition2 = (v / gait_frequency * (2 - abs(phase_p2 - 2)) / 2, 0, end_trajectory_generator(phase_p2))
#         else:
#             addition2 = (0, 0, 0)
#         # cmd0 = q.ik(0, base_pos + addition1, 'shoulder')
#         # cmd1 = q.ik(1, base_pos + addition2, 'shoulder')
#         # cmd2 = q.ik(2, base_pos + addition2, 'shoulder')
#         # cmd3 = q.ik(3, base_pos + addition1, 'shoulder')
#
#         cmd0 = q.ik(0, base_pos, 'shoulder')
#         cmd1 = q.ik(1, base_pos, 'shoulder')
#         cmd2 = q.ik(2, base_pos, 'shoulder')
#         cmd3 = q.ik(3, base_pos, 'shoulder')
#         tq = e.step(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
#         np.set_printoptions(2)
#         # print(*tq, sep='\n', end='\n\n')
#         phase += phase_step
#         # time.sleep(1. / 240.)
#     e.close()
