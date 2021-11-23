from abc import ABC, abstractmethod
import numpy as np

from burl.utils.transforms import Rotation, Rpy


class Sensor(ABC):
    quantity_name = 'quantity'

    def __init__(self, obj, dim):
        self._obj, self._dim = obj, dim
        self._checked = False
        self._observation = None

    @property
    def observation(self):
        return self._observation

    @property
    def observation_dim(self):
        return self._dim

    def observe(self):
        observation = np.asarray(self._on_observe())
        if not self._checked:
            if len(observation.shape) != 1 or observation.shape[0] != self._dim:
                raise RuntimeError('Ambiguous Observation!')
            self._checked = True
        self._observation = observation
        return observation

    @abstractmethod
    def _on_observe(self):
        raise NotImplementedError


class ContactStateSensor(Sensor):
    quantity_name = 'contact_state'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_contact_states()


class OrientationSensor(Sensor):
    quantity_name = 'orientation'

    def __init__(self, obj, dim=4, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.orientation


class OrientationRpySensor(Sensor):
    quantity_name = 'orientation_rpy'

    def __init__(self, obj, dim=3, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return Rpy.from_quaternion(self._obj.orientation)


class GravityVectorSensor(Sensor):
    quantity_name = 'gravity_vector'

    def __init__(self, obj, dim=3, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return Rotation.from_quaternion(self._obj.orientation).Z


class TwistSensor(Sensor):
    quantity_name = 'twist'

    def __init__(self, obj, dim=6, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_twist()


class JointPosHistorySensor(Sensor):
    quantity_name = 'joint_pos_history'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)
        self.moment = kwargs.get('moment')

    def _on_observe(self):
        pass


class JointVelHistorySensor(Sensor):
    quantity_name = 'joint_vel_history'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)
        self.moment = kwargs.get('moment')

    def _on_observe(self):
        pass


class MotorEncoder(Sensor):
    quantity_name = 'motor_angle'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_position()


class MotorEncoderDiff(Sensor):
    quantity_name = 'motor_angle_diff'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_velocity()


class MotorEncoderDiff2(Sensor):
    quantity_name = 'motor_angle_diff2'

    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_acceleration()
