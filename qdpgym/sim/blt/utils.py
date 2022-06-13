class JointInfo(object):
    def __init__(self, joint_info: tuple):
        self._info = joint_info

    idx = property(lambda self: self._info[0])
    name = property(lambda self: self._info[1].decode("UTF-8"))
    type = property(lambda self: self._info[2])
    q_idx = property(lambda self: self._info[3])
    u_idx = property(lambda self: self._info[4])
    damping = property(lambda self: self._info[6])
    friction = property(lambda self: self._info[7])
    limits = property(lambda self: (self._info[8], self._info[9]))
    max_force = property(lambda self: self._info[10])
    max_vel = property(lambda self: self._info[11])
    link_name = property(lambda self: self._info[12].decode("UTF-8"))
    axis = property(lambda self: self._info[13])
    parent_frame_pos = property(lambda self: self._info[14])
    parent_frame_orn = property(lambda self: self._info[15])
    parent_idx = property(lambda self: self._info[16])

    import pybullet as pyb
    joint_types = {pyb.JOINT_REVOLUTE: 'rev', pyb.JOINT_PRISMATIC: 'pri', pyb.JOINT_SPHERICAL: 'sph',
                   pyb.JOINT_PLANAR: 'pla', pyb.JOINT_FIXED: 'fix'}

    axis_dict = {(1., 0., 0.): '+X', (0., 1., 0.): '+Y', (0., 0., 1.): '+Z',
                 (-1., 0., 0.): '-X', (0., -1., 0.): '-Y', (0., 0., -1.): '-Z', }


class DynamicsInfo(object):
    def __init__(self, dynamics_info: tuple):
        self._info = dynamics_info

    mass = property(lambda self: self._info[0])
    lateral_fric = property(lambda self: self._info[1])
    inertia = property(lambda self: self._info[2])
    inertial_pos = property(lambda self: self._info[3])
    inertial_orn = property(lambda self: self._info[4])
    restitution = property(lambda self: self._info[5])
    rolling_fric = property(lambda self: self._info[6])
    spinning_fric = property(lambda self: self._info[7])
    damping = property(lambda self: self._info[8])
    stiffness = property(lambda self: self._info[9])
    body_type = property(lambda self: self._info[10])
    collision_margin = property(lambda self: self._info[11])
