from burl.transforms import Rotation, Quaternion, Rpy
from burl.bc import Sensor


class ContactStateSensor(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_contact_states()


class OrientationSensor(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_pose().orientation


class OrientationRpySensor(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return Rpy.from_quaternion(self._obj.get_pose().orientation)


class GravityVectorSensor(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return Rotation.from_quaternion(self._obj.get_pose().orientation).Z


class MotorEncoder(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_position()


class MotorEncoderDiff(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_velocity()


class MotorEncoderDiff2(Sensor):
    def __init__(self, obj, dim, **kwargs):
        super().__init__(obj, dim)

    def _on_observe(self):
        return self._obj.get_acceleration()
