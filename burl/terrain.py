import pybullet

import pybullet_data

_set_search_path = False





def make_plane(env=pybullet, *args, **kwargs):
    global _set_search_path
    if not _set_search_path:
        env.setAdditionalSearchPath(pybullet_data.getDataPath())
        _set_search_path = True
    p = env.loadURDF("plane.urdf")
    env.changeDynamics(p, -1, lateralFriction=5.0)
    return p
