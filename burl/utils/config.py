from attr import ib, attrs

@attrs
class PhysicsParam(object):
    self_collision_enabled = ib(type=bool, default=False)
    latency = ib(type=float, default=0.)
    on_rack = ib(type=bool, default=False)
    joint_friction = ib(type=float, default=0.025)
    foot_lateral_friction = ib(type=float, default=1.)
    foot_spinning_friction = ib(type=float, default=0.2)
    foot_restitution = ib(type=float, default=0.3)


@attrs
class SimParam(PhysicsParam):
    action_frequency = ib(type=float, default=50.)
    sim_frequency = ib(type=float, default=400.)
    execution_frequency = ib(type=float, default=400.)
    max_sim_iterations = ib(type=int, default=8000)  # 20s


@attrs
class RenderParam(object):
    rendering_enabled = ib(type=bool, default=False)
    sleeping_enabled = ib(type=bool, default=False)
    camera_distance = ib(type=float, default=1.0)
    camera_yaw = ib(type=float, default=0)
    camera_pitch = ib(type=float, default=-30)
    render_width = ib(type=int, default=480)
    render_height = ib(type=int, default=360)
    egl_rendering = ib(type=bool, default=False)

@attrs
class AlgParam(object):
    num_steps_per_env = ib(type=int, default=24)
    num_learning_epochs = ib(type=int, default=1)
    num_mini_batches = ib(type=int, default=1)
    clip_param = ib(type=float, default=0.2)
    gamma = ib(type=float, default=0.995)
    lam = ib(type=float, default=0.95)
    value_loss_coef = ib(type=float, default=1.0)
    entropy_coef = ib(type=float, default=0.0)
    learning_rate = ib(type=float, default=1e-4)
    max_grad_norm = ib(type=float, default=1.0)
    use_clipped_value_loss = ib(type=bool, default=True)
    schedule = ib(type=str, default='fixed')
    desired_kl = ib(type=float, default=0.01)


@attrs
class TrainParam(object):
    max_iterations = ib(type=int, default=1500)
    num_envs = ib(type=int, default=4)
    init_noise_std = ib(type=float, default=0.05)
    save_interval = ib(type=int, default=50)
    device = ib(type=str, default='cuda')


class TaskParam(object):
    def __init__(self):
        self.sim_param = SimParam()
        self.train_param = TrainParam()
        self.render_param = RenderParam()
