import os
import sys
from datetime import datetime
from typing import Union

import burl
from burl.utils import make_part, g_cfg, log_warn

__all__ = ['get_timestamp', 'str2time', 'find_log', 'find_log_remote', 'find_csc', 'find_huzhou',
           'parse_args', 'update_cfg_from_args', 'log_dir', 'teacher_log_dir', 'student_log_dir']

log_dir = os.path.join(os.path.dirname(burl.pkg_path), 'log')
teacher_log_dir = os.path.join(log_dir, 'teacher')
student_log_dir = os.path.join(log_dir, 'student')


def get_timestamp(timestamp: Union[int, float] = None) -> str:
    datetime_ = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
    return datetime_.strftime('%y-%m-%d_%H-%M-%S')


def str2time(time_str: str):
    try:
        if '#' in time_str:
            time_str, *_ = time_str.split('#')
        return datetime.strptime(time_str, '%y-%m-%d_%H-%M-%S')
    except ValueError:
        return datetime(1900, 1, 1)


def _get_folder_with_specific_time(folders, time_=None):
    if not time_:
        return folders[0]
    else:
        for folder in folders:
            if ''.join(folder.split('_')[1].split('-')).startswith(str(time_)):
                return folder
        else:
            raise RuntimeError(f'Record with time `{time_}` not found')


def _get_folder_with_specific_run_name(folders, run_name):
    for folder in folders:
        if folder.split('#')[1].split('@')[0] == run_name:
            return folder
    else:
        raise RuntimeError(f'Record with run name `{run_name}` not found')


def _get_folder_with_specific_run_id(folders, run_id):
    for folder in folders:
        if folder.split('@')[1] == run_id:
            return folder
    else:
        raise RuntimeError(f'Record with run id `{run_id}` not found')


def _get_folder(folders, run_name: str = None, run_id: str = None, time_: Union[int, str] = None):
    if run_name:
        return _get_folder_with_specific_run_name(folders, run_name)
    elif run_id:
        return _get_folder_with_specific_run_id(folders, run_id)
    else:
        return _get_folder_with_specific_time(folders, time_)


def _get_model_of_specific_epoch(models, fmt, epoch):
    prefix, suffix = fmt.split('*')
    final_epoch = max(int(m.removeprefix(prefix).removesuffix(suffix)) for m in models if m)
    if epoch:
        assert epoch <= final_epoch, f'Epoch {epoch} does not exist, max {final_epoch}'
    else:
        epoch = final_epoch
    return prefix + str(epoch) + suffix


def find_log(run_name: str = None, run_id: str = None, time_: Union[int, str] = None,
             epoch: int = None, log_dir=teacher_log_dir, fmt='model_*.pt'):
    folders = sorted(os.listdir(log_dir), key=str2time, reverse=True)
    folder = os.path.join(log_dir, _get_folder(folders, run_name, run_id, time_))
    return os.path.join(folder, _get_model_of_specific_epoch(os.listdir(folder), fmt, epoch))


def find_log_remote(host: str, port: int = None, log_dir='teacher-student/log/teacher', fmt='model_*.pt',
                    run_name: str = None, run_id: str = None, time_: Union[int, str] = None, epoch: int = None):
    """Find and download model file from remote"""
    ssh = f'ssh -p {port} {host}' if port else f'ssh {host}'
    scp = f'scp -P {port} {host}' if port else f'scp {host}'
    print(cmd := f'{ssh} ls {log_dir}')
    remote_logs = os.popen(cmd).read().split('\n')
    remote_logs.remove('')
    folders = sorted(remote_logs, key=str2time, reverse=True)
    print('remote log items: ', *folders)
    folder = _get_folder(folders, run_name, run_id, time_)
    dst_dir = os.path.join(log_dir, folder).replace('\\', '/')

    print(cmd := f'{ssh} ls {dst_dir}')
    models = os.popen(cmd).read().split('\n')
    model_name = _get_model_of_specific_epoch(models, fmt, epoch)
    remote_log = os.path.join(log_dir, folder, model_name)
    local_log_dir = os.path.join('log', 'remote-' + folder)
    if not os.path.exists(os.path.join(local_log_dir, model_name)):
        os.makedirs(local_log_dir, exist_ok=True)
        print(f'downloading model file')
        if os.system(f'{scp}:{remote_log} {local_log_dir}'):
            raise RuntimeError('scp failed')
    return os.path.join(local_log_dir, model_name)


find_csc = make_part(find_log_remote, host='csc')
find_huzhou = make_part(find_log_remote, host='huzhou')
find_wuzhen = make_part(find_log_remote, host='wuzhen')

def parse_args(args: list[str] = None):
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
        assert name.startswith('--'), f"arg `{name}` does not start with --"
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


def update_cfg_from_args():
    abbrs = {'num_iters': 'num_iterations',
             'rand_dyn': 'random_dynamics',
             'centralized': 'use_centralized_curriculum'}
    for name, value in parse_args():
        if name == 'mp_train':
            g_cfg.use_wandb = True
            g_cfg.use_mp = True
            g_cfg.rendering = False
            log_warn(f'wandb: on')
            log_warn(f'multi-process: on')
        elif name == 'cuda' or name == 'cpu':
            g_cfg.device = name
        elif name == 'on_plain':
            g_cfg.trn_type = 'plain'
            log_warn(f'terrain type: plain')
        else:
            if name in abbrs:
                name = abbrs[name]
            if not hasattr(g_cfg, name):
                raise RuntimeError(f"g_cfg has no attribute named '{name}'")
            setattr(g_cfg, name, value)
            value = getattr(g_cfg, name)
            if isinstance(value, bool):
                log_warn(f"{name}: {'on' if value else 'off'}")
            else:
                log_warn(f'{name}: {type(value).__name__} -> {value}')
