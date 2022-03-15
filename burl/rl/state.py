import numpy as np

zero = np.zeros
one = np.ones


class ArrayAttr(object):
    def __setattr__(self, key, value):
        super().__setattr__(key, np.asarray(value, dtype=float))


zero3, zero4, zero8 = (0.,) * 3, (0.,) * 4, (0.,) * 8
zero12, zero24, zero36 = (0.,) * 12, (0.,) * 24, (0.,) * 36


class ObservationBase(object):
    dim: int
    biases: np.ndarray
    weights: np.ndarray
    _init = False

    def __init__(self):
        if not self._init:
            self._init = True
            self._wb_init()
            inst = self.__class__()
            if not inst.to_array().shape == self.biases.shape == self.weights.shape:
                raise RuntimeError(f'{self.__class__.__name__} Shape Check Failed')

    @classmethod
    def _wb_init(cls):
        if cls._init:
            return
        cls._init = True

    def to_array(self) -> np.ndarray:
        raise NotImplementedError

    def standard(self):
        return (self.to_array() - self.biases) * self.weights


def get_robot_type():
    from burl.sim import A1, AlienGo
    return AlienGo


def get_base_frequency():
    from burl.sim import TgStateMachine
    return TgStateMachine.base_frequency


class StateSnapshot(ObservationBase):
    dim = 36
    _init = False

    def __init__(self):
        super().__init__()
        self.command = zero3
        self.gravity_vector = zero3
        self.base_linear = zero3
        self.base_angular = zero3
        self.joint_pos = zero12
        self.joint_vel = zero12

    @classmethod
    def _wb_init(cls):
        if cls._init:
            return
        cls._init = True
        super()._wb_init()
        command_bias, command_weight = zero3, (1.,) * 3
        gravity_vector_bias, gravity_vector_weight = (0., 0., .99), (5., 5., 20.)
        base_linear_bias, base_linear_weight = zero3, (2.,) * 3
        base_angular_bias, base_angular_weight = zero3, (2.,) * 3
        joint_pos_bias, joint_pos_weight = get_robot_type().STANCE_POSTURE, (2.,) * 12
        joint_vel_bias, joint_vel_weight = zero12, (0.5, 0.4, 0.3) * 4
        cls.biases = np.concatenate((
            command_bias,
            gravity_vector_bias,
            base_linear_bias,
            base_angular_bias,
            joint_pos_bias,
            joint_vel_bias
        ))

        cls.weights = np.concatenate((
            command_weight,
            gravity_vector_weight,
            base_linear_weight,
            base_angular_weight,
            joint_pos_weight,
            joint_vel_weight
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


class ProprioObservation(StateSnapshot):
    dim = 60 + 73
    _init = False

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
    def _wb_init(cls):
        if cls._init:
            return
        cls._init = True
        StateSnapshot._wb_init()
        joint_prev_pos_err_bias, joint_prev_pos_err_weight = zero12, (6.5, 4.5, 3.5) * 4
        ftg_phases_bias, ftg_phases_weight = zero8, (1.,) * 8
        ftg_frequencies_bias, ftg_frequencies_weight = (get_base_frequency(),) * 4, (100.,) * 4
        joint_pos_err_his_bias, joint_pos_err_his_weight = zero24, (5.,) * 24
        joint_vel_his_bias, joint_vel_his_weight = zero24, (0.5, 0.4, 0.3) * 8
        joint_pos_target_bias, joint_pos_target_weight = get_robot_type().STANCE_POSTURE, (2.,) * 12
        joint_prev_pos_target_bias, joint_prev_pos_target_weight = get_robot_type().STANCE_POSTURE, (2.,) * 12
        base_frequency_bias, base_frequency_weight = (get_base_frequency(),), (1.,)
        cls.biases = np.concatenate((
            StateSnapshot.biases,
            joint_prev_pos_err_bias,
            ftg_phases_bias,
            ftg_frequencies_bias,
            joint_pos_err_his_bias,
            joint_vel_his_bias,
            joint_pos_target_bias,
            joint_prev_pos_target_bias,
            base_frequency_bias,
        ))

        cls.weights = np.concatenate((
            StateSnapshot.weights,
            joint_prev_pos_err_weight,
            ftg_phases_weight,
            ftg_frequencies_weight,
            joint_pos_err_his_weight,
            joint_vel_his_weight,
            joint_pos_target_weight,
            joint_prev_pos_target_weight,
            base_frequency_weight,
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
    dim = 82
    _init = False

    def __init__(self):
        super().__init__()
        self.terrain_scan = zero36
        self.terrain_normal = zero12
        self.contact_states = zero12
        self.foot_contact_forces = zero12
        self.foot_friction_coeffs = zero4
        self.external_force = zero3
        self.external_torque = zero3

    @classmethod
    def _wb_init(cls):
        if cls._init:
            return
        cls._init = True
        terrain_scan_bias, terrain_scan_weight = zero36, (10.,) * 36
        terrain_normal_bias, terrain_normal_weight = (0., 0., 0.98) * 4, (5., 5., 10.) * 4
        contact_states_bias, contact_states_weight = (0.5,) * 12, (2.,) * 12
        foot_contact_forces_bias, foot_contact_forces_weight = (0., 0., 30.) * 4, (0.01, 0.01, 0.02) * 4
        foot_friction_coeffs_bias, foot_friction_coeffs_weight = zero4, (1.,) * 4
        external_force_bias, external_force_weight = zero3, (0.1,) * 3
        external_torque_bias, external_torque_weight = zero3, (0.4, 0.2, 0.2)
        cls.biases = np.concatenate((
            terrain_scan_bias,
            terrain_normal_bias,
            contact_states_bias,
            foot_contact_forces_bias,
            foot_friction_coeffs_bias,
            external_force_bias,
            external_torque_bias
        ))

        cls.weights = np.concatenate((
            terrain_scan_weight,
            terrain_normal_weight,
            contact_states_weight,
            foot_contact_forces_weight,
            foot_friction_coeffs_weight,
            external_force_weight,
            external_torque_weight
        ))

    def to_array(self):
        return np.concatenate((
            self.terrain_scan,
            self.terrain_normal,
            self.contact_states,
            self.foot_contact_forces,
            self.foot_friction_coeffs,
            self.external_force,
            self.external_torque
        ))


class ExtendedObservation(ExteroObservation, ProprioObservation):
    dim = ExteroObservation.dim + ProprioObservation.dim
    _init = False

    def __init__(self):
        ExteroObservation.__init__(self)
        ProprioObservation.__init__(self)

    @classmethod
    def _wb_init(cls):
        if cls._init:
            return
        cls._init = True
        ExteroObservation._wb_init()
        ProprioObservation._wb_init()
        cls.biases = np.concatenate((ExteroObservation.biases, ProprioObservation.biases))
        cls.weights = np.concatenate((ExteroObservation.weights, ProprioObservation.weights))

    def to_array(self):
        return np.concatenate((ExteroObservation.to_array(self),
                               ProprioObservation.to_array(self)))


class Action(ArrayAttr):
    dim = 12

    def __init__(self):
        self.foot_pos_residuals = zero12

    biases = np.zeros(12)
    weights = np.array((0.25, 0.25, 0.15) * 4)

    @classmethod
    def from_array(cls, arr: np.ndarray):
        arr = arr * cls.weights + cls.biases
        action = Action()
        action.foot_pos_residuals = arr
        return action


# class Action(ArrayAttr):
#     dim = 16
#
#     def __init__(self):
#         self.leg_frequencies = zero4
#         self.foot_pos_residuals = zero12
#
#     biases = np.zeros(16)
#     # weights = np.concatenate((
#     #     (0.01,) * 4, (0.1, 0.1, 0.025) * 4
#     # ))
#
#     weights = np.concatenate((
#         (0.,) * 4, (0.25, 0.25, 0.15) * 4
#     ))
#
#     @classmethod
#     def from_array(cls, arr: np.ndarray):
#         arr = arr * cls.weights + cls.biases
#         action = Action()
#         action.leg_frequencies = arr[:4]
#         action.foot_pos_residuals = arr[4:]
#         return action


class JointStates(ArrayAttr):
    def __init__(self, position=None, velocity=None, reaction_force=None, torque=None):
        self.position, self.velocity = position, velocity
        self.reaction_force, self.torque = reaction_force, torque


class Pose(ArrayAttr):
    def __init__(self, position=None, orientation=None, rpy=None):
        self.position, self.orientation = position, orientation
        self.rpy = rpy


class Twist(ArrayAttr):
    def __init__(self, linear=None, angular=None):
        self.linear, self.angular = linear, angular

    def __iter__(self):
        return (self.linear, self.angular).__iter__()

    def __str__(self):
        return str(np.concatenate([self.linear, self.angular]))


class BaseState(object):
    def __init__(self, pose=None, twist=None, twist_Base=None):
        self.pose: Pose = pose
        self.twist: Twist = twist
        self.twist_Base: Twist = twist_Base

    def __iter__(self):
        return (self.pose, self.twist).__iter__()

    def __str__(self):
        return f'pose: {str(self.pose)}, twist: {str(self.twist)}'


class ContactStates(np.ndarray):
    def __new__(cls, matrix):
        return np.asarray(matrix, dtype=float)


class FootStates(ArrayAttr):
    def __init__(self, positions=None, orientations=None, forces=None):
        self.positions, self.orientations = positions, orientations
        self.forces = forces


class ObservationRaw(object):
    def __init__(self, base_state=None, joint_states=None, foot_states=None, contact_states=None):
        self.base_state: BaseState = base_state
        self.joint_states: JointStates = joint_states
        self.foot_states: FootStates = foot_states
        self.contact_states: ContactStates = contact_states

    def __str__(self):
        return str(self.base_state) + '\n' + str(self.contact_states)


if __name__ == '__main__':
    ExteroObservation()
    ProprioObservation()
    ExtendedObservation()
    ExteroObservation()
