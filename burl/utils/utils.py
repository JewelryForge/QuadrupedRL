import os
import sys
import time
from datetime import datetime


class make_cls(object):
    """
    Predefine some init parameters of a class without creating an instance.
    """

    def __init__(self, cls, *args, **kwargs):
        self.cls, self.args, self.kwargs = cls, args, kwargs

    def __call__(self, *args, **kwargs):
        kwargs.update(self.kwargs)
        return self.cls(*self.args, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.cls, item)


def _make_class(cls, **properties):
    class TemporaryClass(cls):
        def __init__(*args, **kwargs):
            properties.update(kwargs)
            super().__init__(*args, **properties)

    return TemporaryClass


class MfTimer(object):
    """
    Multifunctional Timer. Typical usage example:

    with MfTimer() as timer:
        do_something()
    print(timer.time_spent)
    """

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

    import pybullet as pyb
    joint_types = {pyb.JOINT_REVOLUTE: 'rev', pyb.JOINT_PRISMATIC: 'pri', pyb.JOINT_SPHERICAL: 'sph',
                   pyb.JOINT_PLANAR: 'pla', pyb.JOINT_FIXED: 'fix'}

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


def timestamp() -> str:
    return datetime.now().strftime('%y-%m-%d_%H-%M-%S')


def str2time(time_str: str):
    try:
        if '#' in time_str:
            time_str, *_ = time_str.split('#')
        return datetime.strptime(time_str, '%y-%m-%d_%H-%M-%S')
    except ValueError:
        return datetime(1900, 1, 1)


def _get_folder_with_specific_time(folders, time_stamp=None):
    if not time_stamp:
        return folders[0]
    else:
        for folder in folders:
            if ''.join(folder.split('_')[1].split('-')).startswith(str(time_stamp)):
                return folder
        else:
            raise RuntimeError(f'Record with time {time_stamp} not found')


def find_log(log_dir='log', fmt='model_*.pt', time=None, epoch=None):
    folders = sorted(os.listdir(log_dir), key=str2time, reverse=True)
    folder = os.path.join(log_dir, _get_folder_with_specific_time(folders, time))
    prefix, suffix = fmt.split('*')
    final_epoch = max(int(m.removeprefix(prefix).removesuffix(suffix)) for m in os.listdir(folder))
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    return os.path.join(folder, prefix + f'{epoch}' + suffix)


def find_log_remote(host='jewel@61.153.52.71', port=10022,
                    log_dir='teacher-student/log', fmt='model_*.pt',
                    time=None, epoch=None):
    """Find and download model file from remote"""
    print(cmd := f'ssh -p {port} {host} ls {log_dir}')
    remote_logs = os.popen(cmd).read().split('\n')
    remote_logs.remove('')
    folders = sorted(remote_logs, key=str2time, reverse=True)
    print('remote log items: ', *folders)
    folder = _get_folder_with_specific_time(folders, time)
    dst_dir = os.path.join(log_dir, folder).replace('\\', '/')

    print(cmd := f'ssh -p {port} {host} ls {dst_dir}')
    models = os.popen(cmd).read().split('\n')
    prefix, suffix = fmt.split('*')
    final_epoch = max(int(m.removeprefix(prefix).removesuffix(suffix)) for m in models if m)
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    model_name = prefix + f'{epoch}' + suffix
    remote_log = os.path.join(log_dir, folder, model_name)
    local_log_dir = os.path.join('log', 'remote-' + folder)
    if not os.path.exists(os.path.join(local_log_dir, model_name)):
        os.makedirs(local_log_dir, exist_ok=True)
        print(f'downloading model file')
        if os.system(f'scp -P {port} {host}:{remote_log} {local_log_dir}'):
            raise RuntimeError('scp failed')
    return os.path.join(local_log_dir, model_name)


def parse_args(args=None):
    """
    Parse args from input or sys.argv, yield pairs of name and value.
    Supported argument formats:
        --<ARG_NAME>=<ARG_VALUE>  ->  ARG_NAME, ARG_VALUE;
        --<ARG_NAME> <ARG_VALUE>  ->  ARG_NAME, ARG_VALUE;
        --<ARG_NAME>  ->  ARG_NAME, True;
        --no-<ARG_NAME>  ->  ARG_NAME, False.
    """
    if not args:
        args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        name = args[idx]
        assert name.startswith('--'), ValueError(f"{name} doesn't start with --")
        name = name.removeprefix('--')
        if '=' in name:
            name, value = name.split('=')
        else:
            if idx < len(args) - 1 and not args[idx + 1].startswith('--'):
                value = args[(idx := idx + 1)]
            elif name.startswith('no-'):
                name, value = name.removeprefix('no-'), False
            else:
                value = True
        idx += 1
        name = name.replace('-', '_')
        yield name, value


if __name__ == '__main__':
    print(*parse_args())
