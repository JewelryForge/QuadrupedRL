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


class ObservationBase(object):
    dim: int
    biases: np.ndarray
    weights: np.ndarray

    @classmethod
    def init(cls):
        cls._class_init()
        inst = cls()
        if not inst.to_array().shape == cls.biases.shape == cls.weights.shape:
            raise RuntimeError(f'{cls.__name__} Shape Check Failed')

    @classmethod
    def _class_init(cls):
        pass

    def to_array(self) -> np.ndarray:
        raise NotImplementedError

    def standard(self):
        return (self.to_array() - self.biases) * self.weights


class SimplifiedObservation(ObservationBase):
    dim = 36
    command_bias, command_weight = zero3, rp(1., 3)
    gravity_vector_bias, gravity_vector_weight = (0., 0., .998), rp(10., 3)
    base_linear_bias, base_linear_weight = zero3, rp(2., 3)
    base_angular_bias, base_angular_weight = zero3, rp(2., 3)
    joint_pos_bias, joint_pos_weight = None, rp(2., 12)
    joint_vel_bias, joint_vel_weight = zero12, (0.5, 0.4, 0.3) * 4

    def __init__(self):
        self.command = zero3
        self.gravity_vector = zero3
        self.base_linear = zero3
        self.base_angular = zero3
        self.joint_pos = zero12
        self.joint_vel = zero12

    @classmethod
    def _class_init(cls):
        cls.biases = np.concatenate((
            cls.command_bias,
            cls.gravity_vector_bias,
            cls.base_linear_bias,
            cls.base_angular_bias,
            cls.joint_pos_bias,
            cls.joint_vel_bias
        ))

        cls.weights = np.concatenate((
            cls.command_weight,
            cls.gravity_vector_weight,
            cls.base_linear_weight,
            cls.base_angular_weight,
            cls.joint_pos_weight,
            cls.joint_vel_weight
        ))

    def to_array(self):
        return np.concatenate((
            self.command,
            self.gravity_vector,
            self.base_linear,
            self.base_angular,
            self.joint_pos,
            self.joint_vel
        ))


class ProprioObservation(SimplifiedObservation):
    dim = 60 + 73
    joint_vel_bias, joint_vel_weight = zero12, (0.5, 0.4, 0.3) * 4
    joint_prev_pos_err_bias, joint_prev_pos_err_weight = zero12, (6.5, 4.5, 3.5) * 4
    ftg_phases_bias, ftg_phases_weight = zero8, rp(1., 8)
    ftg_frequencies_bias, ftg_frequencies_weight = None, rp(100., 4)
    joint_pos_err_his_bias, joint_pos_err_his_weight = zero24, rp(5., 24)
    joint_vel_his_bias, joint_vel_his_weight = zero24, (0.5, 0.4, 0.3) * 8
    joint_pos_target_bias, joint_pos_target_weight = None, rp(2., 12)
    joint_prev_pos_target_bias, joint_prev_pos_target_weight = None, rp(2., 12)
    base_frequency_bias, base_frequency_weight = None, (1.,)

    def __init__(self):
        super().__init__()
        self.joint_prev_pos_err = zero12
        self.ftg_phases = zero8
        self.ftg_frequencies = zero4
        self.joint_pos_err_his = zero24
        self.joint_vel_his = zero24
        self.joint_pos_target = zero12
        self.joint_prev_pos_target = zero12
        self.base_frequency = (0.,)

    @classmethod
    def _class_init(cls):
        SimplifiedObservation._class_init()
        cls.biases = np.concatenate((
            SimplifiedObservation.biases,
            cls.joint_prev_pos_err_bias,
            cls.ftg_phases_bias,
            cls.ftg_frequencies_bias,
            cls.joint_pos_err_his_bias,
            cls.joint_vel_his_bias,
            cls.joint_pos_target_bias,
            cls.joint_prev_pos_target_bias,
            cls.base_frequency_bias,
        ))

        cls.weights = np.concatenate((
            SimplifiedObservation.weights,
            cls.joint_prev_pos_err_weight,
            cls.ftg_phases_weight,
            cls.ftg_frequencies_weight,
            cls.joint_pos_err_his_weight,
            cls.joint_vel_his_weight,
            cls.joint_pos_target_weight,
            cls.joint_prev_pos_target_weight,
            cls.base_frequency_weight,
        ))

    def to_array(self):
        return np.concatenate((
            super().to_array(),
            self.joint_prev_pos_err,
            self.ftg_phases,
            self.ftg_frequencies,
            self.joint_pos_err_his,
            self.joint_vel_his,
            self.joint_pos_target,
            self.joint_prev_pos_target,
            self.base_frequency
        ))


class ExteroObservation(ObservationBase):
    dim = 79
    terrain_scan_bias, terrain_scan_weight = zero36, rp(10., 36)
    terrain_normal_bias, terrain_normal_weight = (0., 0., 0.98) * 4, (5., 5., 10.) * 4
    contact_states_bias, contact_states_weight = rp(0.5, 12), (2.,) * 12
    foot_contact_forces_bias, foot_contact_forces_weight = (0., 0., 30.) * 4, (0.01, 0.01, 0.02) * 4
    foot_friction_coeffs_bias, foot_friction_coeffs_weight = zero4, rp(1., 4)
    external_disturbance_bias, external_disturbance_weight = zero3, rp(0.1, 3)

    def __init__(self):
        self.terrain_scan = zero36
        self.terrain_normal = zero12
        self.contact_states = zero12
        self.foot_contact_forces = zero12
        self.foot_friction_coeffs = zero4
        self.external_disturbance = zero3

    @classmethod
    def _class_init(cls):
        cls.biases = np.concatenate((
            cls.terrain_scan_bias,
            cls.terrain_normal_bias,
            cls.contact_states_bias,
            cls.foot_contact_forces_bias,
            cls.foot_friction_coeffs_bias,
            cls.external_disturbance_bias
        ))

        cls.weights = np.concatenate((
            cls.terrain_scan_weight,
            cls.terrain_normal_weight,
            cls.contact_states_weight,
            cls.foot_contact_forces_weight,
            cls.foot_friction_coeffs_weight,
            cls.external_disturbance_weight
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

    def __init__(self):
        ExteroObservation.__init__(self)
        ProprioObservation.__init__(self)

    @classmethod
    def _class_init(cls):
        ExteroObservation._class_init()
        ProprioObservation._class_init()
        cls.biases = np.concatenate((ExteroObservation.biases, ProprioObservation.biases))
        cls.weights = np.concatenate((ExteroObservation.weights, ProprioObservation.weights))

    def to_array(self):
        return np.concatenate((ExteroObservation.to_array(self),
                               ProprioObservation.to_array(self)))


class Action:
    dim = 16

    def __init__(self):
        self.leg_frequencies = np.zeros(4)
        self.foot_pos_residuals = np.zeros(12)

    biases = np.zeros(16)
    weights = np.concatenate((
        (0.01,) * 4, (0.1, 0.1, 0.025) * 4
    ))

    @classmethod
    def from_array(cls, arr: np.ndarray):
        arr = arr * cls.weights + cls.biases
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
    s = SimplifiedObservation()
    SimplifiedObservation.joint_pos_bias = rp(0., 12)
    SimplifiedObservation.init()
    print(s.dim)
    print(s.bias.shape, s.scale.shape)
    print(s.to_array().shape)
