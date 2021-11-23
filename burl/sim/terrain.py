import pybullet


def make_plane(env=pybullet, *args, **kwargs):
    p = env.loadURDF("plane.urdf")
    env.changeDynamics(p, -1, lateralFriction=5.0)
    return p
