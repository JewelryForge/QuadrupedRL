from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class Options(object):
    rendering: bool = False
    on_rack: bool = False
    test_mode: bool = False
    use_wandb: bool = True
    add_disturbance: bool = True
    trn_type: str = 'plain'
    tg_init: str = 'symmetric'
    lr_scheduler: str = ''
    task_type: str = 'basic'
    random_dynamics: bool = False
    actuator_net: Union[str, None] = None
    use_centralized_curriculum: bool = False
    aggressive: bool = False


@dataclass
class PhysicsParam(object):
    self_collision_enabled: bool = False
    latency_range: tuple[float, float] = (0., 0.03)
    motor_latencies: tuple[float, float] = (0., 0.)
    joint_friction: float = 0.025
    foot_lateral_friction: float = 0.4
    foot_spinning_friction: float = 0.2
    foot_restitution: float = 0.3
    joint_angle_range: float = 1.0


@dataclass
class SimParam(PhysicsParam):
    action_frequency: int = 50
    sim_frequency: int = 500
    execution_frequency: int = 500
    max_sim_iterations: int = 10000
    use_action_interp: bool = True


@dataclass
class RenderParam(object):
    rendering: bool = False
    gui: bool = False
    sleeping_enabled: bool = False
    time_ratio: float = 1.
    moving_camera: bool = True
    driving_mode: bool = False
    extra_visualization: bool = True
    show_indicators: bool = True
    show_time_ratio: bool = True
    plot_trajectory: bool = False
    single_step_rendering: bool = False
    record: bool = False


@dataclass
class PPOParam(object):
    storage_len: int = 128
    repeat_times: int = 8
    num_mini_batches: int = 1
    clip_ratio: float = 0.2
    gamma: float = 0.99
    lambda_gae: float = 0.95
    value_loss_coeff: float = 1.0
    entropy_coeff: float = 3e-3
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0


@dataclass
class ImitationParam(object):
    model_path: str = ''
    batch_size: int = 2000
    num_steps_each_epoch: int = 2000
    history_len: int = 100


@dataclass
class TrainParam(object):
    num_iterations: int = 10000
    num_envs: int = 16
    init_noise_std: float = 0.1
    save_interval: int = 50
    device: str = 'cuda'
    extero_layer_dims: tuple[int, ...] = (72, 64)
    proprio_layer_dims: tuple[int, ...] = ()
    # action_layer_dims: tuple[int, ...] = (256, 128, 64)
    action_layer_dims: tuple[int, ...] = (512, 256, 128)


@dataclass
class RuntimeParam(object):
    log_dir: str = ''
    run_name: str = ''
    use_mp: bool = True
    rewards_weights: tuple[tuple[str, float], ...] = (('UnifiedLinearReward', 0.1),
                                                      ('YawRateReward', 0.06),
                                                      ('VerticalLinearPenalty', 0.04),
                                                      ('RollPitchRatePenalty', 0.04),
                                                      ('BodyPosturePenalty', 0.04),
                                                      ('FootSlipPenalty', 0.04),
                                                      ('BodyCollisionPenalty', 0.04),
                                                      ('TorquePenalty', 0.01),
                                                      ('JointMotionPenalty', 0.01),
                                                      ('ActionSmoothnessReward', 0.01),
                                                      ('ClearanceOverTerrainReward', 0.02),

                                                      ('AliveReward', 0.0),
                                                      ('LinearVelocityReward', 0.0),
                                                      ('OrthogonalLinearPenalty', 0.0),
                                                      ('CostOfTransportReward', 0.),
                                                      ('BodyHeightReward', 0.),
                                                      ('HipAnglePenalty', 0.),)


@dataclass
class DisturbanceParam(object):
    disturbance_interval_steps: int = 500
    force_magnitude: tuple[float, float] = (20., 20.)  # horizontal vertical
    torque_magnitude: tuple[float, float, float] = (2.5, 5., 5.)  # x y z


class TaskParam(Options, SimParam, RenderParam, TrainParam, ImitationParam,
                PPOParam, DisturbanceParam, RuntimeParam):
    def __init__(self):
        Options.__init__(self)
        SimParam.__init__(self)
        RenderParam.__init__(self)
        TrainParam.__init__(self)
        ImitationParam.__init__(self)
        PPOParam.__init__(self)
        DisturbanceParam.__init__(self)
        RuntimeParam.__init__(self)
        self._init = True

    @property
    def dev(self):
        return self.device

    @property
    def fps(self):
        return self.sim_frequency if self.single_step_rendering else self.action_frequency

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
        if (default_value := getattr(self, key)) is not None:
            if not isinstance(default_value, str) and isinstance(value, str):
                value = eval(value)
            if not isinstance(value, default_type := type(default_value)):
                try:
                    value = default_type(value)
                except ValueError:
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
