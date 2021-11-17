import attr


@attr.s
class RenderParam(object):
    enable_rendering = attr.ib(type=bool, default=True)
    camera_distance = attr.ib(type=float, default=1.0)
    camera_yaw = attr.ib(type=float, default=0)
    camera_pitch = attr.ib(type=float, default=-30)
    render_width = attr.ib(type=int, default=480)
    render_height = attr.ib(type=int, default=360)
    egl_rendering = attr.ib(type=bool, default=False)

# @attr.s
# class SimulationConfig(object):
#     num_solver_iterations