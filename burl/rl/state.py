import numpy as np

zero = np.zeros
one = np.ones


class ArrayAttr(object):
    def __setattr__(self, key, value):
        super().__setattr__(key, np.asarray(value, dtype=float))


def rp(element, times):  # repeat
    return (element,) * times


zero3, zero4, zero8 = rp(0., 3), rp(0., 4), rp(0., 8)
zero12, zero24, zero36 = rp(0., 12), rp(0., 24), rp(0., 36)


class ProprioObservation(object):
    dim = 60 + 73
    command_avg, command_scl = zero3, rp(1., 3)
    gravity_vector_avg, gravity_vector_scl = (0., 0., .998), rp(10., 3)
    base_linear_avg, base_linear_scl = zero3, rp(2., 3)
    base_angular_avg, base_angular_scl = zero3, rp(2., 3)
    joint_pos_avg, joint_pos_scl = None, rp(2., 12)
    joint_vel_avg, joint_vel_scl = zero12, (0.5, 0.4, 0.3) * 4
    joint_prev_pos_err_avg, joint_prev_pos_err_scl = zero12, (6.5, 4.5, 3.5) * 4
    ftg_phases_avg, ftg_phases_scl = zero8, rp(1., 8)
    ftg_frequencies_avg, ftg_frequencies_scl = None, rp(100., 4)
    joint_pos_err_his_avg, joint_pos_err_his_scl = zero24, rp(5., 24)
    joint_vel_his_avg, joint_vel_his_scl = zero24, (0.5, 0.4, 0.3) * 8
    joint_pos_target_avg, joint_pos_target_scl = None, rp(2., 12)
    joint_prev_pos_target_avg, joint_prev_pos_target_scl = None, rp(2., 12)
    base_frequency_avg, base_frequency_scl = None, (1.,)
    offset, scale = None, None

    def __init__(self):
        self.command = zero3
        self.gravity_vector = zero3
        self.base_linear = zero3
        self.base_angular = zero3
        self.joint_pos = zero12
        self.joint_vel = zero12
        self.joint_prev_pos_err = zero12
        self.ftg_phases = zero8
        self.ftg_frequencies = zero4
        self.joint_pos_err_his = zero24
        self.joint_vel_his = zero24
        self.joint_pos_target = zero12
        self.joint_prev_pos_target = zero12
        self.base_frequency = (0.,)

    @classmethod
    def init(cls):
        cls.offset = np.concatenate((
            cls.command_avg,
            cls.gravity_vector_avg,
            cls.base_linear_avg,
            cls.base_angular_avg,
            cls.joint_pos_avg,
            cls.joint_vel_avg,
            cls.joint_prev_pos_err_avg,
            cls.ftg_phases_avg,
            cls.ftg_frequencies_avg,
            cls.joint_pos_err_his_avg,
            cls.joint_vel_his_avg,
            cls.joint_pos_target_avg,
            cls.joint_prev_pos_target_avg,
            cls.base_frequency_avg,
        ))

        cls.scale = np.concatenate((
            cls.command_scl,
            cls.gravity_vector_scl,
            cls.base_linear_scl,
            cls.base_angular_scl,
            cls.joint_pos_scl,
            cls.joint_vel_scl,
            cls.joint_prev_pos_err_scl,
            cls.ftg_phases_scl,
            cls.ftg_frequencies_scl,
            cls.joint_pos_err_his_scl,
            cls.joint_vel_his_scl,
            cls.joint_pos_target_scl,
            cls.joint_prev_pos_target_scl,
            cls.base_frequency_scl,
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
            self.ftg_frequencies,
            self.joint_pos_err_his,
            self.joint_vel_his,
            self.joint_pos_target,
            self.joint_prev_pos_target,
            self.base_frequency
        ))


class ExteroObservation(object):
    dim = 79
    terrain_scan_avg, terrain_scan_scl = zero36, rp(10., 36)
    terrain_normal_avg, terrain_normal_scl = (0., 0., 0.98) * 4, (5., 5., 10.) * 4
    contact_states_avg, contact_states_scl = rp(0.5, 12), (2.,) * 12
    foot_contact_forces_avg, foot_contact_forces_scl = (0., 0., 30.) * 4, (0.01, 0.01, 0.02) * 4
    foot_friction_coeffs_avg, foot_friction_coeffs_scl = zero4, rp(1., 4)
    external_disturbance_avg, external_disturbance_scl = zero3, rp(0.1, 3)
    offset, scale = None, None

    def __init__(self):
        self.terrain_scan = zero36
        self.terrain_normal = zero12
        self.contact_states = zero12
        self.foot_contact_forces = zero12
        self.foot_friction_coeffs = zero4
        self.external_disturbance = zero3

    @classmethod
    def init(cls):
        cls.offset = np.concatenate((
            cls.terrain_scan_avg,
            cls.terrain_normal_avg,
            cls.contact_states_avg,
            cls.foot_contact_forces_avg,
            cls.foot_friction_coeffs_avg,
            cls.external_disturbance_avg
        ))

        cls.scale = np.concatenate((
            cls.terrain_scan_scl,
            cls.terrain_normal_scl,
            cls.contact_states_scl,
            cls.foot_contact_forces_scl,
            cls.foot_friction_coeffs_scl,
            cls.external_disturbance_scl
        ))

    def to_array(self):
        return np.concatenate((
            self.terrain_scan,
            self.terrain_normal,
            self.contact_states,
            self.foot_contact_forces,
            self.foot_friction_coeffs,
            self.external_disturbance
        ))


class ExtendedObservation(ExteroObservation, ProprioObservation):
    dim = ExteroObservation.dim + ProprioObservation.dim
    offset, scale = None, None

    def __init__(self):
        ExteroObservation.__init__(self)
        ProprioObservation.__init__(self)

    @classmethod
    def init(cls):
        ExteroObservation.init()
        ProprioObservation.init()
        cls.offset = np.concatenate((ExteroObservation.offset, ProprioObservation.offset))
        cls.scale = np.concatenate((ExteroObservation.scale, ProprioObservation.scale))

    def to_array(self):
        return np.concatenate((ExteroObservation.to_array(self),
                               ProprioObservation.to_array(self)))

    def standard(self):
        return (self.to_array() - self.offset) * self.scale


class Action:
    dim = 16

    def __init__(self):
        self.leg_frequencies = zero(4)
        self.foot_pos_residuals = zero(12)

    offset = np.zeros(16)
    scale = np.concatenate((
        (0.01,) * 4, (0.1, 0.1, 0.025) * 4
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


class FootStates(ArrayAttr):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('positions', 'orientations', 'forces'), args))
        self.positions = kwargs.get('positions', None)
        self.orientations = kwargs.get('orientations', None)
        self.forces = kwargs.get('forces', None)


class ObservationRaw(object):
    def __init__(self, *args, **kwargs):
        kwargs.update(zip(('base_state', 'joint_states', 'foot_states',
                           'foot_forces', 'contact_states', 'contact_info'), args))
        self.base_state: BaseState = kwargs.get('base_state', None)
        self.joint_states: JointStates = kwargs.get('joint_states', None)
        self.foot_states: FootStates = kwargs.get('foot_states', None)
        # self.foot_forces: np.ndarray = kwargs.get('foot_forces', None)
        # self.foot_positions: np.ndarray = kwargs.get('foot_positions', None)
        self.contact_states: ContactStates = kwargs.get('contact_states', None)
        # self.contact_info: tuple = kwargs.get('contact_info', None)

    def __str__(self):
        return str(self.base_state) + '\n' + str(self.contact_states)


if __name__ == '__main__':
    s = ProprioObservation()
    print(s.dim)
    print(s.offset.shape, s.scale.shape)
    print(s.to_array().shape)
