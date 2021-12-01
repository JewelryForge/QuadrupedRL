import torch

from burl.rl.state import ExtendedObservation, Action
from burl.utils.utils import timestamp

class PhysicsParam(object):
    def __init__(self):
        self.self_collision_enabled = False
        self.latency = 0.
        self.on_rack = False
        self.joint_friction = 0.025
        self.foot_lateral_friction = 1.0
        self.foot_spinning_friction = 0.2
        self.foot_restitution = 0.3


class SimParam(PhysicsParam):
    def __init__(self):
        super().__init__()
        self.action_frequency = 50.0
        self.sim_frequency = 400.
        self.execution_frequency = 400.
        self.max_sim_iterations = 8000  # 20s


class RenderParam(object):
    def __init__(self):
        self.rendering_enabled = False
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


class TerrainParam(object):
    def __init__(self):
        self.plain = False
        self.trn_size = 30
        self.trn_downsample = 5
        self.trn_roughness = 0.1
        self.trn_resolution = 0.05
        self.trn_offset = (0., 0., 0.)


class TaskParam(SimParam, RenderParam, TrainParam, TerrainParam):
    def __init__(self):
        SimParam.__init__(self)
        RenderParam.__init__(self)
        TrainParam.__init__(self)
        TerrainParam.__init__(self)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key == 'device':
            try:
                global g_dev
                g_dev = g_cfg.device
            except NameError:
                pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


g_cfg = TaskParam()
g_dev = g_cfg.device
