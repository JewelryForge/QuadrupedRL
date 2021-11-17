import time

import gym
import numpy as np
import pybullet
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client

from burl.bc import Observable
from burl.config import RenderParam
from burl.quadruped import QuadrupedSim
from burl.sensors import OrientationRpySensor, GravityVectorSensor
from burl.terrain import make_plane
from burl.utils import make_cls


def trajectory_generator(k, h=0.12):  # 0.2 in paper
    # k = 2 * normalize(phase) / np.pi
    if 0 < k <= 1:
        return h * (-2 * k ** 3 + 3 * k ** 2)
    if 1 < k <= 2:
        return h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4)
    return 0


class LocomotionEnv(gym.Env, Observable):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self._render_cfg = kwargs.get('render_config', RenderParam())
        self._gui = self._render_cfg.enable_rendering
        if self._gui:
            self._env = bullet_client.BulletClient(pybullet.GUI)
            self._init_rendering()
        else:
            self._env = bullet_client.BulletClient(pybullet.DIRECT)
        if False:
            self._env = pybullet
        _make_sensors = kwargs.get('make_sensors', [])
        super().__init__(make_sensors=_make_sensors)
        _make_robot = kwargs.get('make_robot', QuadrupedSim)
        self._robot: QuadrupedSim = _make_robot(sim_env=self._env)
        self._subordinates.append(self._robot)

        self._terrain_generator = kwargs.get('terrain_generator', make_plane)
        self._random_state, self._random_seed = self.seed(kwargs.get('seed', None))
        self._sim_frequency = kwargs.get('sim_frequency', 240)
        self._action_frequency = kwargs.get('action_frequency', 120)
        assert self._sim_frequency >= self._robot.frequency >= self._action_frequency

        self._terrain = self._terrain_generator(self._env)
        self._set_physics_parameters()
        self._robot.update_observation()

        self._num_action_repeats = int(self._sim_frequency / self._action_frequency)
        self._num_execution_repeats = int(self._sim_frequency / self._robot.frequency)
        print(f'Action Repeats for {self._num_action_repeats} time(s)')
        print(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        self._sim_step_counter = 0

    def _on_update_observation(self):
        pass

    @property
    def robot(self):
        return self._robot

    @property
    def action_space(self):
        return spaces.Box(*np.array(self._robot.action_limits), dtype=np.float64)

    @property
    def observation_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,), dtype=np.float64)

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
                self._robot.apply_command(action)
            self._env.stepSimulation()
            if update_execution:
                self._robot.update_observation()
            self._sim_step_counter += 1
        if not self._robot.is_safe():
            self.reset()
        if self._gui:
            self._check_render_options()

    def reset(self, **kwargs):
        completely_reset = kwargs.get('completely_reset', False)
        if completely_reset:
            raise NotImplementedError
        self._robot.reset()

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
        pass

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

        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        time_to_sleep = self._num_action_repeats / self._sim_frequency - time_spent
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        # Also keep the previous orientation of the camera set by the user.
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


if __name__ == '__main__':
    # for i in np.linspace(0, 2 * np.pi):
    #     print(trajectory_generator(i))

    v = 0.3
    make_robot = make_cls(QuadrupedSim, on_rack=False, make_sensors=[OrientationRpySensor, GravityVectorSensor])
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
        # cmd0 = q.ik(0, base_pos + addition1, 'shoulder')
        # cmd1 = q.ik(1, base_pos + addition2, 'shoulder')
        # cmd2 = q.ik(2, base_pos + addition2, 'shoulder')
        # cmd3 = q.ik(3, base_pos + addition1, 'shoulder')

        cmd0 = q.ik(0, base_pos, 'shoulder')
        cmd1 = q.ik(1, base_pos, 'shoulder')
        cmd2 = q.ik(2, base_pos, 'shoulder')
        cmd3 = q.ik(3, base_pos, 'shoulder')
        tq = e.step(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
        phase += phase_step
        time.sleep(1. / 240.)
