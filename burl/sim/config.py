import attr


@attr.s
class RenderParam(object):
    rendering_enabled = attr.ib(type=bool, default=True)
    sleeping_enabled = attr.ib(type=bool, default=False)
    camera_distance = attr.ib(type=float, default=1.0)
    camera_yaw = attr.ib(type=float, default=0)
    camera_pitch = attr.ib(type=float, default=-30)
    render_width = attr.ib(type=int, default=480)
    render_height = attr.ib(type=int, default=360)
    egl_rendering = attr.ib(type=bool, default=False)


@attr.s
class TaskParam(object):
    action_frequency = attr.ib(type=float, default=50.)
    sim_frequency = attr.ib(type=float, default=400.)
    execution_frequency = attr.ib(type=float, default=400.)
    num_steps_per_env = attr.ib(type=int, default=24)  # per iteration
    max_iterations = attr.ib(type=int, default=1500)  # number of policy updates
    num_envs = attr.ib(type=int, default=4)
    save_interval = attr.ib(type=int, default=50)
