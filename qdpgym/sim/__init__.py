import os

import qdpgym

pkg_path = os.path.dirname(os.path.abspath(__file__))
rsc_dir = os.path.join(pkg_path, 'resources')


def is_bullet_available():
    try:
        import pybullet
    except ModuleNotFoundError:
        return False
    return True


def is_mujoco_available():
    try:
        import mujoco
        import dm_control
    except ModuleNotFoundError:
        return False
    return True


if qdpgym.sim_engine == qdpgym.Sim.MUJOCO:
    assert is_mujoco_available(), 'dm_control is not installed'
    from .mjc.quadruped import Aliengo
    from .mjc.env import QuadrupedEnv
    from .mjc.terrain import *
    from .mjc.hooks import *
else:
    assert is_bullet_available(), 'pybullet is not installed'
    from .blt.quadruped import Aliengo
    from .blt.env import QuadrupedEnv
    from .blt.terrain import *
    from .blt.hooks import *

from .abc import *
from .app import Application
