import numpy as np
import torch

from burl.utils import timestamp


class Options(object):
    def __init__(self):
        self.rendering = False
        self.on_rack = False
        self.test_mode = False
        self.use_wandb = True
        self.add_disturbance = True
        self.trn_type = 'plain'
        self.tg_init = 'fixed'
        self.schedule = 'fixed'
        self.random_dynamics = False
        self.actuator_net = None


class PhysicsParam(object):
    def __init__(self):
        self.self_collision_enabled = False
        self.motor_latencies = (0., 0.)
        self.joint_friction = 0.025
        self.foot_lateral_friction = 0.4
        self.foot_spinning_friction = 0.2
        self.foot_restitution = 0.3
        self.joint_angle_range = 1.0


class SimParam(PhysicsParam):
    def __init__(self):
        super().__init__()
        self.action_frequency = 50
        self.sim_frequency = 500
        self.execution_frequency = 500
        self.max_sim_iterations = 10000
        self.use_action_interpolation = True


class RenderParam(object):
    def __init__(self):
        self.rendering = False
        self.sleeping_enabled = False
        self.time_ratio = 1.
        self.moving_camera = True
        self.extra_visualization = True
        self.plot_trajectory = False
        self.single_step_rendering = False


class PPOParam(object):
    def __init__(self):
        self.storage_len = 256
        self.num_learning_epochs = 4
        self.num_mini_batches = 1
        self.clip_param = 0.2
        self.gamma = 0.995
        self.lambda_ = 0.95
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.
        self.learning_rate = 1e-4
        self.max_grad_norm = 1.0
        self.use_clipped_value_loss = True
        self.desired_kl = 0.01


class TrainParam(object):
    def __init__(self):
        self.num_iterations = 10000
        self.num_envs = 8
        self.init_noise_std = (.05,) * 4 + (.1, .05, .05) * 4
        self.save_interval = 50
        self.device = torch.device('cuda')
        self.extero_layer_dims = (72, 64)
        self.proprio_layer_dims = ()
        self.action_layer_dims = (256, 128, 64)


class RuntimeParam(object):
    def __init__(self):
        self.log_dir = f'log/{timestamp()}'
        self.run_name = None
        self.use_mp = True
        self.rewards_weights = (('LinearVelocityReward', 0.06),
                                ('YawRateReward', 0.06),
                                ('VerticalLinearPenalty', 0.04),
                                ('AliveReward', 0.12),
                                ('OrthogonalLinearPenalty', 0.04),
                                ('RollPitchRatePenalty', 0.04),
                                ('BodyPosturePenalty', 0.04),
                                ('FootSlipPenalty', 0.04),
                                # ('TrivialStridePenalty', 0.06),
                                # ('TorqueGradientPenalty', 0.04),
                                ('ClearanceOverTerrainReward', 0.04),
                                ('BodyCollisionPenalty', 0.04),
                                ('TorquePenalty', 0.01),
                                ('CostOfTransportReward', 0.),
                                ('BodyHeightReward', 0.),
                                ('HipAnglePenalty', 0.),)


class TerrainParam(object):
    def __init__(self):
        self.trn_size = 30
        self.trn_slope = np.pi / 24
        self.trn_downsample = 15
        self.trn_roughness = 0.1
        self.trn_resolution = 0.05
        self.trn_offset = (0., 0., 0.)


class TerrainCurriculumParam(object):
    def __init__(self):
        self.combo_threshold = 5
        self.miss_threshold = 3
        self.difficulty_step = 0.01
        self.max_difficulty = 0.3
        self.distance_threshold = (2.5, 5.0)


class DisturbanceParam(object):
    def __init__(self):
        self.disturbance_interval_steps = 500
        self.horizontal_force_bounds = (0., 20.)
        self.vertical_force_bounds = (0., 20.)


class TaskParam(Options, SimParam, RenderParam, TrainParam, PPOParam, DisturbanceParam,
                TerrainParam, TerrainCurriculumParam, RuntimeParam):
    def __init__(self):
        Options.__init__(self)
        SimParam.__init__(self)
        RenderParam.__init__(self)
        TrainParam.__init__(self)
        PPOParam.__init__(self)
        DisturbanceParam.__init__(self)
        TerrainParam.__init__(self)
        TerrainCurriculumParam.__init__(self)
        RuntimeParam.__init__(self)
        self._init = True

    @property
    def dev(self):
        return self.device

    def test_profile(self):
        self.test_mode = True
        self.rendering = True
        self.use_mp = False
        self.use_wandb = False
        self.sleeping_enabled = True

    def slow_down_rendering(self, time_ratio=0.1):
        self.time_ratio = time_ratio
        self.single_step_rendering = True

    def update(self, kv: dict):
        for k, v in kv.items():
            setattr(self, k, v)

    def __setattr__(self, key, value):
        if not hasattr(self, '_init'):
            return object.__setattr__(self, key, value)
        if key == 'device':
            value = torch.device(value)
        if (default_value := getattr(self, key)) is not None:
            if not isinstance(default_value, str) and isinstance(value, str):
                value = eval(value)
            if not isinstance(value, type(default_value)):
                raise RuntimeError(f'Value type {value} of key {key} is not {type(default_value)}')
        return object.__setattr__(self, key, value)

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            setattr(cls, '_instance', super().__new__(cls))
        return getattr(cls, '_instance')


g_cfg = TaskParam()


def to_dev(tensor: torch.Tensor, *tensors):
    if not tensors:
        return tensor.to(g_cfg.dev)
    else:
        return tuple([t.to(g_cfg.dev) for t in [tensor, *tensors]])
