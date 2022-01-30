import math
import os
import sys
import time
from datetime import datetime

import numpy as np

M_PI = np.pi
M_2_PI = 2 * M_PI


def ang_norm(x):  # [-pi, pi)
    return x - ((x + M_PI) // M_2_PI) * M_2_PI


def unit(x) -> np.ndarray:
    return np.asarray(x) / math.hypot(*x)


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def vec_cross(vec3_1, vec3_2) -> np.ndarray:
    """
    A much faster alternative for np.cross.
    """
    (x1, y1, z1), (x2, y2, z2) = vec3_1, vec3_2
    return np.array((y1 * z2 - y2 * z1, z1 * x2 - z2 * x1, x1 * y2 - x2 * y1))


def truncate(value: float, lower: float, upper: float) -> float:
    return upper if value > upper else lower if value < lower else value


def safe_asin(value):
    return math.asin(truncate(value, -1., 1.))


def safe_acos(value):
    return math.acos(truncate(value, -1., 1.))


def included_angle(vec1, vec2) -> float:
    return safe_acos(np.dot(unit(vec1), unit(vec2)))


def tuple_compact_string(_tuple, precision=1):
    fmt = f'.{precision}f'
    return '(' + ' '.join(f'{f:{fmt}}' for f in _tuple) + ')'


class make_cls(object):
    """
    Predefine some init parameters of a class without creating an object.
    """

    def __init__(self, cls, **properties):
        self.cls, self.properties = cls, properties

    def __call__(self, *args, **kwargs):
        properties = self.properties.copy()
        properties.update(**kwargs)
        return self.cls(*args, **properties)

    def __getattr__(self, item):
        return getattr(self.cls, item)


def _make_class(cls, **properties):
    class TemporaryClass(cls):
        def __init__(*args, **kwargs):
            properties.update(kwargs)
            super().__init__(*args, **properties)

    return TemporaryClass


def get_const_variables(cls):
    var = {}
    for key, item in cls.__dict__:
        if key.isupper():
            var[key] = item
    return var


class MfTimer:
    def __init__(self, func=None, *args, **kwargs):
        if func:
            self.start()
            func(*args, **kwargs)
            self.end()

    def start(self):
        self._start_time = time.time()

    def __enter__(self):
        self._start_time = time.time()
        return self

    def end(self):
        self._end_time = time.time()
        return self.time_spent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()

    @property
    def time_spent(self):
        return self._end_time - self._start_time


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

    import pybullet as p
    joint_types = {p.JOINT_REVOLUTE: 'rev', p.JOINT_PRISMATIC: 'pri', p.JOINT_SPHERICAL: 'sph',
                   p.JOINT_PLANAR: 'pla', p.JOINT_FIXED: 'fix'}

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


def timestamp():
    return datetime.now().strftime('%b%d_%H-%M-%S')


def str2time(time_str):
    try:
        return datetime.strptime(time_str, '%b%d_%H-%M-%S')
    except ValueError:
        return datetime(1900, 1, 1)


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def find_log(log_dir='log', time=None, epoch=None):
    folders = sorted(os.listdir(log_dir), key=str2time, reverse=True)
    if not time:
        folder = folders[0]
    else:
        for folder in folders:
            if ''.join(folder.split('_')[1].split('-')).startswith(str(time)):
                break
        else:
            raise RuntimeError(f'Record with time {time} not found')
    folder = os.path.join(log_dir, folder)
    final_epoch = max(int(m.removeprefix('model_').removesuffix('.pt'))
                      for m in os.listdir(folder) if m.startswith('model'))
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    return os.path.join(folder, f'model_{epoch}.pt')


def find_log_remote(host='jewel@61.153.52.71', port=10022, log_dir='teacher-student-dev/log', time=None, epoch=None):
    print(f'ssh {host} -p {port} ls {log_dir}')
    remote_logs = os.popen(f'ssh {host} -p {port} ls {log_dir}').read().split('\n')
    remote_logs.remove('')
    folders = sorted(remote_logs, key=str2time, reverse=True)
    print('remote log items: ', *folders)
    if not time:
        folder = folders[0]
    else:
        for folder in folders:
            if ''.join(folder.split('_')[1].split('-')).startswith(str(time)):
                break
        else:
            raise RuntimeError(f'Record with time {time} not found, all {folders}')
    dst_dir = os.path.join(log_dir, folder).replace('\\', '/')
    print(f'ssh {host} -p {port} ls {dst_dir}')
    models = os.popen(f'ssh {host} -p {port} ls {dst_dir}').read().split('\n')
    final_epoch = max(int(m.removeprefix('model_').removesuffix('.pt'))
                      for m in models if m.startswith('model'))
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    model_name = f'model_{epoch}.pt'
    remote_log = os.path.join(log_dir, folder, model_name)
    local_log_dir = os.path.join('log', 'remote-' + folder)
    if not os.path.exists(os.path.join(local_log_dir, model_name)):
        os.makedirs(local_log_dir, exist_ok=True)
        print(f'downloading model file')
        if os.system(f'scp -P {port} {host}:{remote_log} {local_log_dir}'):
            raise RuntimeError('scp failed')
    return os.path.join(local_log_dir, model_name)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, kwargs = iter(argv), {}
    while True:
        try:
            name = next(args)
        except StopIteration:
            break
        assert name.startswith('--'), ValueError(f"{name} doesn't start with --")
        name = name.removeprefix('--')
        if '=' in name:
            name, value = name.split('=')
        else:
            try:
                value = next(args)
                assert not value.startswith('--')
            except (StopIteration, AssertionError):
                raise RuntimeError(f"Parameter '{name}' has no corresponding value")
        name = name.replace('-', '_')
        kwargs[name] = value
    return kwargs


if __name__ == '__main__':
    print(truncate(-0.6, -1, 1))
