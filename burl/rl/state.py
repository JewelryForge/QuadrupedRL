import numpy as np


# class PlainObservation:
#     def __init__(self):


class ArrayAttr(object):
    def __setattr__(self, key, value):
        super().__setattr__(key, np.asarray(value, dtype=float))


class ProprioceptiveObservation(ArrayAttr):
    dim = 60

    def __init__(self):
        self.command = np.zeros(3)
        self.gravity_vector = np.zeros(3)
        self.base_linear = np.zeros(3)
        self.base_angular = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_prev_pos_err = np.zeros(12)  # if needed
        self.ftg_phases = np.zeros(8)
        self.ftg_frequencies = np.zeros(4)

    offset = (
        0., 0., 0.,
        0., 0., 1.,
        0., 0., 0.,
        0., 0., 0.,

    )

    def to_array(self):
        return np.concatenate([
            self.command,
            self.gravity_vector,
            self.base_linear,
            self.base_angular,
            self.joint_pos,
            self.joint_vel,
            self.joint_prev_pos_err,
            self.ftg_phases,
            self.ftg_frequencies
        ])


class Observation(ProprioceptiveObservation):
    dim = ProprioceptiveObservation.dim + 73

    def __init__(self):
        super(Observation, self).__init__()
        self.base_frequency = np.array((1.25,))
        self.joint_pos_err_his = np.zeros(24)
        self.joint_vel_his = np.zeros(24)
        self.joint_pos_target = np.zeros(12)
        self.joint_prev_pos_target = np.zeros(12)

    def to_array(self):
        return np.concatenate([
            super().to_array(),
            self.base_frequency,
            self.joint_pos_err_his,
            self.joint_vel_his,
            self.joint_pos_target,
            self.joint_prev_pos_target
        ])


class PrivilegedInformation(object):
    dim = 79

    def __init__(self):
        self.terrain_scan = np.zeros(36)
        self.terrain_normal = np.array([0, 0, 1] * 4)
        self.contact_states = np.zeros(12)
        self.foot_contact_forces = np.zeros(12)
        self.foot_friction_coeffs = np.zeros(4)
        self.external_disturbance = np.zeros(3)

    def to_array(self):
        return np.concatenate([
            self.terrain_scan,
            self.terrain_normal,
            self.contact_states,
            self.foot_contact_forces,
            self.foot_friction_coeffs,
            self.external_disturbance
        ])


class ExtendedObservation(Observation, PrivilegedInformation):
    dim = Observation.dim + PrivilegedInformation.dim

    def __init__(self):
        Observation.__init__(self)
        PrivilegedInformation.__init__(self)

    def to_array(self):
        return np.concatenate([
            Observation.to_array(self),
            PrivilegedInformation.to_array(self)
        ])


class Action:
    dim = 16

    def __init__(self):
        self.leg_frequencies = np.zeros(4)

        self.foot_pos_residuals = np.zeros(12)

    def from_array(self, arr: np.ndarray):
        self.leg_frequencies = arr[:4]
        self.foot_pos_residuals = arr[4:]


zero = np.zeros


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
        kwargs.update(zip(('base_state', 'joint_states', 'contact_states'), args))
        self.base_state: BaseState = kwargs.get('base_state', None)
        self.joint_states: JointStates = kwargs.get('joint_states', None)
        self.contact_states: ContactStates = kwargs.get('contact_states', None)

    def __str__(self):
        return str(self.base_state) + '\n' + str(self.contact_states)


if __name__ == '__main__':
    s = MotorState([0, 1, 2])
    s.__dict__.update({'a': 3})
    print(s.a)
