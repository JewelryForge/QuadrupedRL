import random
from typing import Type

from burl.rl.curriculum import CURRICULUM_PROTOTYPE, CentralizedCurriculum
from burl.rl.reward import *
from burl.sim.plugins import Plugin, StatisticsCollector, InfoRenderer, VideoRecorder
from burl.sim.terrain import Terrain, Plain, Hills, Steps, Slope, Stairs
from burl.utils import g_cfg, make_part

__all__ = ['BasicTask', 'RandomLinearCmdTask', 'RandomCmdTask', 'get_task', 'CentralizedTask']


class BasicTask(RewardRegistry):
    def __init__(self, env, cmd=(0.0, 0.0, 0.0)):
        super().__init__(np.asarray(cmd), env, env.robot)
        for reward, weight in g_cfg.rewards_weights:
            self.add_reward(reward, weight)
        self.set_coeff(0.5)

        self.plugins: list[Plugin] = []
        self.plugin_utils = {}
        if g_cfg.test_mode:
            self.load_plugin(StatisticsCollector())
        if g_cfg.record:
            self.load_plugin(VideoRecorder())
            g_cfg.show_time_ratio = False
        if g_cfg.rendering:
            self.load_plugin(InfoRenderer(g_cfg.extra_visualization, g_cfg.show_time_ratio,
                                          g_cfg.show_indicators, g_cfg.driving_mode,
                                          g_cfg.moving_camera, g_cfg.sleeping_enabled,
                                          g_cfg.single_step_rendering))

    @property
    def cmd(self):
        return self._cmd

    @cmd.setter
    def cmd(self, cmd):
        self._cmd = np.asarray(cmd)

    def getCommandObservation(self):
        if (self._cmd[:2] == 0.).all():
            return np.concatenate(((0.,), self._cmd))
        else:
            return np.concatenate(((1.,), self._cmd))

    env = property(lambda self: self._env)
    robot = property(lambda self: self._robot)

    def make_terrain(self, terrain_type: str) -> Terrain:
        if terrain_type == 'plain':
            terrain_obj = Plain()
        elif terrain_type == 'hills':
            terrain_obj = Hills.make(30, 0.1, (0.4, 20), (0.02, 1))
        elif terrain_type == 'slope':
            terrain_obj = Slope.make(20, 0.05, 0.17, 2.0)
        elif terrain_type == 'steps':
            terrain_obj = Steps.make(20, 0.05, 1.0, 0.4)
        elif terrain_type == 'stairs':
            terrain_obj = Stairs.make(20, 0.05, 0.15, 0.3)
        elif terrain_type == 'curriculum':
            return self.plugin_utils['generate_terrain'](self.env.client)
        else:
            raise RuntimeError(f'Unknown terrain type {terrain_type}')
        terrain_obj.spawn(self._env.client)
        return terrain_obj

    def load_plugin(self, plugin: Plugin):
        self.plugins.append(plugin)
        for plg_util in plugin.utils:
            self.plugin_utils[plg_util] = getattr(plugin, plg_util)

    def on_init(self):
        """Called back after env init"""
        for plg in self.plugins:
            plg.on_init(self, self._robot, self._env)

    def on_sim_step(self):
        """Called back after every simulation step"""
        for plg in self.plugins:
            plg.on_sim_step(self, self._robot, self._env)

    def on_step(self):
        """Called back after every env.step"""
        info = {}
        for plg in self.plugins:
            if plg_info := plg.on_step(self, self._robot, self._env):
                info.update(plg_info)
        return info

    def reset(self):
        """Called back before env resets"""
        for plg in self.plugins:
            plg.on_reset(self, self._robot, self._env)

    def is_regularly_finished(self):
        if self._env.sim_step >= g_cfg.max_sim_iterations:
            return True
        x, y, _ = self._env.robot.getBasePosition()
        if self._env.terrain.out_of_range(x, y):
            return True
        return False

    def is_failed(self):
        r, _, _ = self._robot.rpy
        safety_h = self._env.getTerrainBasedHeightOfRobot()
        h_lb, h_ub = self._robot.STANCE_HEIGHT * 0.5, self._robot.STANCE_HEIGHT * 1.5
        if (safety_h < h_lb or safety_h > h_ub or r < -np.pi / 3 or r > np.pi / 3 or
                self._robot.getBaseContactState()):
            return True
        # joint_diff = self._robot.getJointPositions() - self._robot.STANCE_POSTURE
        # if any(joint_diff > g_cfg.joint_angle_range) or any(joint_diff < -g_cfg.joint_angle_range):
        #     return True
        return False


# class RandomLeftRightTask(BasicTask):
#     def __init__(self, env):
#         self.update_interval = 1500
#         self.last_update = 0
#         self.last_cmd = 0
#         super().__init__(env, (0., 1., 0.))
#
#     def reset(self):
#         self.last_update = 0
#         self._cmd = np.array((0., 1., 0.))
#         super().reset()
#
#     def on_step(self):
#         if self._env.sim_step >= self.last_update + self.update_interval:
#             self._cmd = np.array((0., 1., 0.) if self.last_cmd else (0., -1., 0.))
#             self.last_cmd = 1 - self.last_cmd
#             self.last_update = self._env.sim_step
#         super().on_step()


class RandomLinearCmdTask(BasicTask):
    """Randomly updates linear command"""

    def __init__(self, env, seed=None):
        random.seed(seed)
        self.stop_prob = 0.2
        self.forward_prob = 0.05
        self.interval_range = (500, 5000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = -1
        super().__init__(env)

    def random_cmd(self):
        # if random.random() < self.stop_prob:
        #     return np.array((0., 0., 0.))
        yaw = random.uniform(0, 2 * np.pi)
        return np.array((math.cos(yaw), math.sin(yaw), 0))

    def reset(self):
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = -1
        self._cmd = np.array((0., 0., 0.))
        super().reset()

    def on_step(self):
        if self.last_update == -1 or self._env.sim_step >= self.last_update + self.update_interval:
            self._cmd = self.random_cmd()
            self.last_update = self._env.sim_step
            self.update_interval = random.uniform(*self.interval_range)
        return super().on_step()


class RandomCmdTask(RandomLinearCmdTask):
    """Randomly updates command"""

    def random_cmd(self):
        angular_cmd = random.choice((-1., 0, 0, 1.))
        if (rand := random.random()) < self.stop_prob:
            return np.array((0., 0., angular_cmd))
        elif rand < self.stop_prob + self.forward_prob:
            return np.array((1., 0., angular_cmd))
        else:
            yaw = random.uniform(0, math.tau)
            return np.array((math.cos(yaw), math.sin(yaw), angular_cmd))
        # return np.array((math.cos(yaw), math.sin(yaw), clip(random.gauss(0, 0.5), -1, 1)))


class RandomVelocityTask(RandomCmdTask):
    def random_cmd(self):
        cmd = super().random_cmd()
        cmd[:2] *= random.uniform(0., 1.)
        return cmd


class CentralizedTask(object):
    """A wrapper of Task class for centralized curricula"""

    def __init__(self):
        self.crm_protos: list[CURRICULUM_PROTOTYPE] = []
        aggressive = g_cfg.test_mode or g_cfg.aggressive
        buffer_len = g_cfg.num_envs
        if g_cfg.use_centralized_curriculum:
            from burl.rl.curriculum import CentralizedDisturbanceCurriculum, CentralizedTerrainCurriculum
            if g_cfg.add_disturbance:
                crm_obj = CentralizedDisturbanceCurriculum(buffer_len=buffer_len, aggressive=aggressive)
                self.crm_protos.append(crm_obj)
            if g_cfg.trn_type == 'curriculum':
                crm_obj = CentralizedTerrainCurriculum(buffer_len=buffer_len, aggressive=aggressive)
                self.crm_protos.append(crm_obj)
        else:
            raise NotImplementedError
            # from burl.rl.curriculum import DisturbanceCurriculum, TerrainCurriculum
            # if g_cfg.add_disturbance:
            #     self.crm_protos.append(DisturbanceCurriculum(aggressive))
            # if g_cfg.trn_type == 'curriculum':
            #     self.crm_protos.append(TerrainCurriculum(aggressive))

    def spawner(self, task_class: Type[BasicTask], args=(), **kwargs):
        return make_part(self.make_distribution, task_class, args=args, **kwargs)

    def make_distribution(self, task_class: Type[BasicTask], env, args=(), **kwargs):
        task_inst = task_class(env, *args, **kwargs)
        for crm in self.crm_protos:
            task_inst.load_plugin(crm.make_distribution())
        return task_inst

    def update_curricula(self):
        for crm in self.crm_protos:
            if isinstance(crm, CentralizedCurriculum):
                crm.summarize()


def get_task(task_type: str):
    if task_type == 'basic':
        return BasicTask
    elif task_type == 'randLn':
        return RandomLinearCmdTask
    elif task_type == 'randCmd':
        return RandomCmdTask
    elif task_type == 'randVel':
        return RandomVelocityTask
    # elif task_type == 'randLR':
    #     return RandomLeftRightTask
    else:
        raise RuntimeError(f"Unknown task type '{task_type}'")
