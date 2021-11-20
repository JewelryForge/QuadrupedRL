from abc import ABC, abstractmethod

import numpy as np


# class Observable(ABC):
#     ALLOWED_SENSORS = {}
#
#     def __init__(self, make_sensors=()):
#         self._sensors: list[Sensor] = []
#         self._subordinates: list[Observable] = []
#         self._sensor_interfaces = {}
#         self._make_sensors(make_sensors)
#
#     @property
#     def observation_dim(self):
#         return sum(s.observation_dim for s in self._sensors) + \
#                sum(s.observation_dim for s in self._subordinates)
#
#     @abstractmethod
#     def _on_update_observation(self):
#         pass
#
#     def update_observation(self, recursively=True):
#         self._on_update_observation()
#         observation = self._process_sensors()
#         if not recursively:
#             return np.concatenate([observation, *(sub._read_sensor_caches() for sub in self._subordinates)])
#         return np.concatenate([observation, *(sub.update_observation() for sub in self._subordinates)])
#
#     def _process_sensors(self):
#         if self._sensors:
#             return np.concatenate([s.observe() for s in self._sensors])
#         return np.array([])
#
#     def _read_sensor_caches(self):
#         if self._sensors:
#             return np.concatenate([s.observation for s in self._sensors])
#         return np.array([])
#
#     def _make_sensors(self, make_sensors):
#         for make in make_sensors:
#             try:
#                 cls = make.__closure__[0].cell_contents
#             except AttributeError:
#                 cls = make
#             if cls not in self.ALLOWED_SENSORS:
#                 raise RuntimeError(f'{cls.__name__} Not Supported For {self.__class__.__name__}')
#             try:
#                 self._sensors.append(make(self, self.ALLOWED_SENSORS[cls]))
#             except TypeError:
#                 self._sensors.append(make(self))


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


# class MotorBase(Observable, ABC):
#     def __init__(self, *args, **kwargs):
#         super(MotorBase, self).__init__(*args, **kwargs)
#
#     frequency = None
#
#     @abstractmethod
#     def reset(self):
#         pass
#
#     @abstractmethod
#     def set_command(self, command, *args):
#         pass


class NDArrayBased(np.ndarray):
    def __new__(cls, matrix, skip_check=False):
        matrix = np.asarray(matrix)
        if skip_check:
            return matrix.view(cls)
        if cls.is_valid(matrix):
            return cls.preprocess(matrix).view(cls)
        raise RuntimeError(f'Invalid {cls}')

    @classmethod
    def is_valid(cls, matrix):
        return True

    @classmethod
    def preprocess(cls, matrix):
        return matrix


# class QuadrupedBase(Observable, ABC):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         pass
#
#     frequency = None
#     action_limits = None
#
#     @abstractmethod
#     def reset(self, *args):
#         pass
#
#     @abstractmethod
#     def apply_command(self, motor_commands):
#         pass
#
#     @abstractmethod
#     def ik(self, *args):
#         pass

class Task(ABC):
    def __init__(self, *args, **kwargs):
        self._rewards = []

    def done(self):
        pass

    @abstractmethod
    def __call__(self, info):
        pass


if __name__ == '__main__':
    NDArrayBased([])
