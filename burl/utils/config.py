import torch

from burl.rl.state import ExtendedObservation, Action
from burl.utils import timestamp


class PhysicsParam(object):
    def __init__(self):
        self.self_collision_enabled = False
        self.latency = 0.
        self.on_rack = False
        self.joint_friction = 0.025
        self.foot_lateral_friction = 1.0
        self.foot_spinning_friction = 0.2
        self.foot_restitution = 0.3
        self.joint_angle_range = 1.0
        self.safe_height_range = (0.2, 0.6)


class SimParam(PhysicsParam):
    def __init__(self):
        super().__init__()
        self.local_urdf = '/home/jewel/Workspaces/teacher-student/urdf'
        self.action_frequency = 50.0
        self.sim_frequency = 400.
        self.execution_frequency = 400.
        self.max_sim_iterations = 8000  # 20s


class RenderParam(object):
    def __init__(self):
        self.rendering = False
        self.sleeping_enabled = False
        self.moving_camera = True
        self.extra_visualization = True
        self.egl_rendering = False


class AlgParam(object):
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
        self.schedule = 'fixed'
        self.desired_kl = 0.01


class TrainParam(AlgParam):
    def __init__(self):
        super().__init__()
        self.num_iterations = 10000
        self.num_envs = 8
        self.init_noise_std = 0.05
        self.save_interval = 50
        self.obs_dim = ExtendedObservation.dim
        self.p_obs_dim = ExtendedObservation.dim
        self.action_dim = Action.dim
        self.device = torch.device('cuda')
        self.log_dir = f'log/{timestamp()}'
        self.run_name = None
        self.task_class = None
        self.use_mp = True
        self.use_wandb = True
        self.rewards_weights = None
        self.ip_address = '10.12.120.120'
        self.port = '19996'


class TerrainParam(object):
    def __init__(self):
        self.plain = False
        self.trn_size = 30
        self.trn_downsample = 15
        self.trn_roughness = 0.1
        self.trn_resolution = 0.05
        self.trn_offset = (0., 0., 0.)


class TerrainCurriculumParam(object):
    def __init__(self):
        self.use_trn_curriculum = False
        self.combo_threshold = 5
        self.miss_threshold = 3
        self.difficulty_step = 0.01
        self.max_difficulty = 0.3
        self.distance_threshold = (2.5, 5.0)


class TaskParam(SimParam, RenderParam, TrainParam, TerrainParam, TerrainCurriculumParam):
    def __init__(self):
        SimParam.__init__(self)
        RenderParam.__init__(self)
        TrainParam.__init__(self)
        TerrainParam.__init__(self)
        TerrainCurriculumParam.__init__(self)
        self._init = True

    @property
    def dev(self):
        return self.device

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
