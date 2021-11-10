import numpy as np

# class PlainObservation:
#     def __init__(self):



class ProprioceptiveObservation:
    dim = 60

    def __init__(self):
        self.des_vel_direction = np.zeros(2)
        self.des_orientation = np.zeros(1)
        self.gravity = np.zeros(3)
        self.base_linear = np.zeros(3)
        self.base_angular = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_prev_pos = np.zeros(12)  # if needed
        self.ftg_phases = np.zeros(8)
        self.ftg_freqs = np.zeros(4)

    def make_array(self):
        return np.concatenate([
            self.des_vel_direction,
            self.des_orientation,
            self.gravity,
            self.base_linear,
            self.base_angular,
            self.joint_pos,
            self.joint_vel,
            self.joint_prev_pos,
            self.ftg_phases,
            self.ftg_freqs
        ])


class Observation(ProprioceptiveObservation):
    dim = ProprioceptiveObservation.dim + 73

    def __init__(self):
        super(Observation, self).__init__()
        self.base_freq = np.array(1.25)
        self.joint_pos_err_his = np.zeros(24)
        self.joint_vel_err_his = np.zeros(24)
        self.joint_pos_target = np.zeros(12)
        self.joint_prev_pos_target = np.zeros(12)

    def make_array(self):
        return np.concatenate([
            super().make_array(),
            self.base_freq,
            self.joint_pos_err_his,
            self.joint_vel_err_his,
            self.joint_pos_target,
            self.joint_prev_pos_target
        ])


class PrivilegedInformation:
    dim = 79

    def __init__(self):
        super(PrivilegedInformation, self).__init__()
        self.terrain_scan = np.zeros(36)
        self.terrain_normal = np.zeros(12)
        self.foot_contact_states = np.zeros(4)
        self.foot_contact_forces = np.zeros(12)
        self.foot_friction_coeffs = np.zeros(4)
        self.thigh_contact_states = np.zeros(4)
        self.shank_contact_states = np.zeros(4)
        self.external_disturbance = np.zeros(3)

    def to_array(self):
        return np.concatenate([
            self.terrain_scan,
            self.terrain_normal,
            self.foot_contact_states,
            self.foot_contact_forces,
            self.foot_friction_coeffs,
            self.thigh_contact_states,
            self.shank_contact_states,
            self.external_disturbance
        ])


class Action:
    dim = 12

    def __init__(self):
        self.leg_freqs = np.zeros(4)
        self.foot_pos_residuals = np.zeros(12)

    def from_array(self, arr: np.ndarray):
        self.leg_freqs = arr[:4]
        self.foot_pos_residuals = arr[4:]


if __name__ == '__main__':
    po = PrivilegedInformation()
    print(po.to_array().shape)
