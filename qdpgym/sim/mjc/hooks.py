from typing import Optional

import mujoco as mjlib

from qdpgym import tf
from qdpgym.sim.abc import Hook
from qdpgym.sim.mjc.viewer import ViewerMj


class ViewerMjHook(Hook):
    def __init__(self):
        self._viewer: Optional[ViewerMj] = None

    def init_episode(self, robot, env):
        self._viewer = ViewerMj(env.physics.model.ptr, env.physics.data.ptr)

    def after_substep(self, robot, env):
        perturb = env.get_perturbation()
        if perturb is not None:
            force = perturb[:3]
            magnitude = tf.vnorm(force)
            mat = tf.Rotation.from_zaxis(force / magnitude)
            self._viewer.add_marker(type=mjlib.mjtGeom.mjGEOM_ARROW,
                                    pos=robot.get_base_pos(), mat=mat,
                                    size=[0.01, 0.01, magnitude / 20],
                                    rgba=(1., 0., 0., 1.))
        self._viewer.render()

