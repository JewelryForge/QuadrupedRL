import numpy as np

zero = np.zeros
one = np.ones


class ArrayAttr(object):
    def __setattr__(self, key, value):
        super().__setattr__(key, np.asarray(value, dtype=float))


freq_scale = np.pi / 200


class ProprioceptiveObservation(ArrayAttr):
    dim = 60

    def __init__(self):
        self.command = zero(3)
        self.gravity_vector = zero(3)
        self.base_linear = zero(3)
        self.base_angular = zero(3)
        self.joint_pos = zero(12)
        self.joint_vel = zero(12)
        self.joint_prev_pos_err = zero(12)  # if needed
        self.ftg_phases = zero(8)
        self.ftg_frequencies = zero(4)

    offset = np.concatenate((
        zero(3), (0., 0., 0.998),  # command & gravity_vector
        zero(6), (0, 0.723, -1.445) * 4,  # base twist & joint pos
        zero(12), zero(12),  # joint vel &  joint_prev_pos_err
        zero(8), (1.5,) * 4  # ftg phases & frequencies
    ))

    scale = np.concatenate((
        (1.0,) * 3, (5.0,) * 3,  # command & gravity_vector
        (2.,) * 6, (2.0,) * 12,  # base twist & joint pos
        # FIXME: why joint_prev_pos_err so large
        (0.5, 0.4, 0.3) * 4, (6.5, 4.5, 3.5) * 4,  # joint vel &  joint_prev_pos_err
        (1.5,) * 8, (2.0 / freq_scale,) * 4  # ftg phases & frequencies; latter 400/pi in paper, why?
    ))

    def to_array(self):
        return np.concatenate((
            self.command,
            self.gravity_vector,
            self.base_linear,
            self.base_angular,
            self.joint_pos,
            self.joint_vel,
            self.joint_prev_pos_err,
            self.ftg_phases,
            self.ftg_frequencies
        ))


class Observation(ProprioceptiveObservation):
    dim = ProprioceptiveObservation.dim + 73

    def __init__(self):
        super(Observation, self).__init__()
        self.joint_pos_err_his = zero(24)
        self.joint_vel_his = zero(24)
        self.joint_pos_target = zero(12)
        self.joint_prev_pos_target = zero(12)
        self.base_frequency = np.array((1.25,))

    offset = np.concatenate((
        ProprioceptiveObservation.offset,
        zero(24), zero(24),  # joint_pos_err_his & joint_vel_his
        (0, 0.723, -1.445) * 8,  # joint_pos_target * 2
        (1.25,)  # base_frequency
    ))

    scale = np.concatenate((
        ProprioceptiveObservation.scale,
        (5.,) * 24, (0.5, 0.4, 0.3) * 8,  # joint_pos_err_his & joint_vel_his
        (2.0,) * 24,  # joint_pos_target * 2
        (1,),  # base_frequency NOTICE: differ from paper
    ))

    def to_array(self):
        return np.concatenate((
            super().to_array(),
            self.joint_pos_err_his,
            self.joint_vel_his,
            self.joint_pos_target,
            self.joint_prev_pos_target,
            self.base_frequency
        ))


class PrivilegedInformation(object):
    dim = 79
    TERRAIN_CLIP = 0.25
    FORCE_CLIP = np.array((25, 25, 50) * 4)

    offset = np.concatenate((
        (0.02,) * 36, (0., 0., 0.98) * 4,  # terrain scan & normal
        (0.5, 0.5, 0.5) * 4,  # contact states
        (0, 0, 30.0) * 4,  # contact forces
        (0., 0., 0., 0.), (0., 0., 0.)  # friction & disturbance
    ))

    scale = np.concatenate((
        (10.,) * 36, (5., 5., 10.) * 4,  # terrain scan & normal
        (2,) * 12,  # contact states
        (0.01, 0.01, 0.02) * 4,  # contact forces
        (1.,) * 4, (1.,) * 3  # friction & disturbance
    ))

    def __init__(self):
        self.terrain_scan = zero(36)  # clip
        self.terrain_normal = np.array((0, 0, 1) * 4)  # clip
        self.contact_states = zero(12)
        self.foot_contact_forces = zero(12)
        self.foot_friction_coeffs = zero(4)
        self.external_disturbance = zero(3)

    def to_array(self):
        return np.concatenate((
            np.clip(self.terrain_scan, -self.TERRAIN_CLIP, self.TERRAIN_CLIP),
            self.terrain_normal,
            self.contact_states,
            np.clip(self.foot_contact_forces, -self.FORCE_CLIP, self.FORCE_CLIP),
            self.foot_friction_coeffs,
            self.external_disturbance
        ))


class ExtendedObservation(Observation, PrivilegedInformation):
    dim = Observation.dim + PrivilegedInformation.dim
    offset = np.concatenate((Observation.offset, PrivilegedInformation.offset))
    scale = np.concatenate((Observation.scale, PrivilegedInformation.scale))

    def __init__(self):
        Observation.__init__(self)
        PrivilegedInformation.__init__(self)

    def to_array(self):
        return np.concatenate((
            Observation.to_array(self),
            PrivilegedInformation.to_array(self)
        ))


class Action:
    dim = 16

    def __init__(self):
        self.leg_frequencies = zero(4)
        self.foot_pos_residuals = zero(12)

    offset = np.zeros(16)

    scale = np.concatenate((
        (0.5 * freq_scale,) * 4, (0.1, 0.1, 0.025) * 4
    ))

    @classmethod
    def from_array(cls, arr: np.ndarray):
        arr = arr * cls.scale + cls.offset
        action = Action()
        action.leg_frequencies = arr[:4]
        action.foot_pos_residuals = arr[4:]
        return action


class MotorState(ArrayAttr):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('position', 'velocity', 'acceleration'), args))
        self.position = kwargs.get('position', zero(0))
        self.velocity = kwargs.get('velocity', zero(0))
        self.acceleration = kwargs.get('acceleration', zero(0))


class JointStates(ArrayAttr):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('position', 'velocity', 'reaction_force', 'torque'), args))
        self.position = kwargs.get('position', zero(0))
        self.velocity = kwargs.get('velocity', zero(0))
        self.reaction_force = kwargs.get('reaction_force', zero((0, 6)))
        self.torque = kwargs.get('torque', zero(0))


class Pose(ArrayAttr):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('position', 'orientation'), args))
        self.position = kwargs.get('position', zero(3))
        self.orientation = kwargs.get('orientation', zero(4))

    def __iter__(self):
        return (self.position, self.orientation).__iter__()

    def __str__(self):
        return str(np.concatenate([self.position, self.orientation]))


class Twist(ArrayAttr):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('linear', 'angular'), args))
        self.linear = kwargs.get('linear', zero(3))
        self.angular = kwargs.get('angular', zero(3))

    def __iter__(self):
        return (self.linear, self.angular).__iter__()

    def __str__(self):
        return str(np.concatenate([self.linear, self.angular]))


class BaseState(object):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('pose', 'twist'), args))
        self.pose = kwargs.get('pose', Pose())
        self.twist = kwargs.get('twist', Twist())

    def __iter__(self):
        return (self.pose, self.twist).__iter__()

    def __str__(self):
        return f'pose: {str(self.pose)}, twist: {str(self.twist)}'


class ContactStates(np.ndarray):
    def __new__(cls, matrix):
        return np.asarray(matrix, dtype=float)


class ObservationRaw(object):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('base_state', 'joint_states', 'foot_forces', 'contact_states'), args))
        self.base_state: BaseState = kwargs.get('base_state', None)
        self.joint_states: JointStates = kwargs.get('joint_states', None)
        self.foot_forces: np.ndarray = kwargs.get('foot_forces', None)
        self.contact_states: ContactStates = kwargs.get('contact_states', None)

    def __str__(self):
        return str(self.base_state) + '\n' + str(self.contact_states)


if __name__ == '__main__':
    s = ExtendedObservation()
    print(s.dim)
    print(s.offset.shape, s.scale.shape)
    print(s.to_array().shape)
