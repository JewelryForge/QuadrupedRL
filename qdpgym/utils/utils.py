import functools
import logging
import math
import re
import time
from datetime import datetime
from typing import Sequence, Callable, Optional, Union

import numpy as np
import yaml

__all__ = ['make_part', 'Angle', 'replace_is', 'replace_eq', 'colored_str', 'log',
           'MfTimer', 'get_padded', 'PadWrapper', 'get_timestamp', 'YamlLoader',
           'print_return', 'AutoInc']


class make_part(functools.partial):
    """
    Predefine some init parameters of a class without creating an instance.
    """

    def __getattr__(self, item):
        return getattr(self.func, item)


PI, TAU = math.pi, math.tau


class Angle(object):
    @staticmethod
    def norm(x):
        """normalize an angle to [-pi, pi)"""
        if isinstance(x, np.ndarray):
            return x - ((x + PI) / TAU).astype(int) * TAU
        if -PI <= x < PI:
            return x
        return x - int((x + PI) / TAU) * TAU

    @staticmethod
    def mean(lst: Sequence[float]):
        _sum = last = lst[0]
        for i, a in enumerate(lst):
            if i == 0:
                continue
            last = Angle.near(last, a)
            _sum += last
        return Angle.norm(_sum / len(lst))

    @staticmethod
    def near(x, y):
        if x > y + PI:
            y += TAU
        elif x < y - PI:
            y -= TAU
        return y

    @staticmethod
    def to_deg(rad):
        return rad / math.pi * 180


def get_padded(queue: Sequence, idx):
    if idx < 0:
        if len(queue) >= -idx:
            return queue[idx]
        else:
            return queue[0]
    else:
        if len(queue) > idx:
            return queue[idx]
        else:
            return queue[-1]


class PadWrapper(object):
    def __init__(self, seq: Sequence):
        self.seq = seq
        self.is_empty = not self.seq

    def __getitem__(self, item):
        if self.is_empty:
            raise IndexError('Empty Sequence')
        else:
            return get_padded(self.seq, item)

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f'PadWrapper[{self.seq}]'


def replace_is(seq: list, src, dst):
    for idx, item in enumerate(seq):
        if item is src:
            seq[idx] = dst


def replace_eq(seq: list, src, dst):
    for idx, item in enumerate(seq):
        if item == src:
            seq[idx] = dst


class MfTimer(object):
    """
    Multifunctional Timer. Typical usage example:

    with MfTimer() as timer:
        do_something()
    print(timer.time_spent)
    """

    @classmethod
    def start_now(cls):
        timer = MfTimer()
        timer.start()
        return timer

    @classmethod
    def record(cls, func: Callable, *args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time

    def __init__(self):
        self._start_time = self._end_time = None

    def start(self):
        self._start_time = time.time()

    def __enter__(self):
        self._start_time = time.time()
        return self

    def end(self, verbose=False):
        self._end_time = time.time()
        if verbose:
            print(self.time_spent)
        return self.time_spent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()

    @property
    def time_spent(self):
        return self._end_time - self._start_time


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"


def colored_str(content: str, color: str):
    try:
        color = eval(color.upper())
    except NameError:
        raise RuntimeError(f'No color named {color}')
    return COLOR_SEQ % (30 + color) + content + RESET_SEQ


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': YELLOW,
        'INFO': WHITE,
        'DEBUG': BLUE,
        'ERROR': RED,
        'CRITICAL': MAGENTA,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = True

    def format(self, record):
        lvl = record.levelname
        message = str(record.msg)
        funcName = record.funcName
        if lvl in self.COLORS:
            lvl_colored = COLOR_SEQ % (30 + self.COLORS[lvl]) + lvl + RESET_SEQ
            msg_colored = COLOR_SEQ % (30 + self.COLORS[lvl]) + message + RESET_SEQ
            funcName_colored = COLOR_SEQ % (30 + self.COLORS[lvl]) + funcName + RESET_SEQ
            record.levelname = lvl_colored
            record.msg = msg_colored
            record.funcName = funcName_colored
        return super().format(record)


class _Log(object):
    logger: Optional[logging.Logger] = None

    @classmethod
    def init_logger(cls, name='logger',
                    log_level: int = logging.INFO,
                    log_fmt='[%(asctime)s] %(message)s',
                    date_fmt='%b%d %H:%M:%S',
                    log_dir=None,
                    client_ip='127.0.0.1'):
        np.set_printoptions(3, linewidth=10000, suppress=True)
        # torch.set_printoptions(linewidth=10000, profile='short')
        cls.logger = logging.getLogger(name)
        cls.logger.setLevel(log_level)
        # if log_dir:
        #     os.makedirs(log_dir)
        #     fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        #     fh.setLevel(logging.DEBUG)
        #     cls.logger.addHandler(fh)
        # soh = SocketHandler(client_ip, 19996)
        # soh = SocketHandler('10.12.120.120', 19996)
        # soh.setFormatter(logging.Formatter())
        # cls.logger.addHandler(soh)
        formatter = ColoredFormatter(log_fmt, date_fmt)
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(formatter)
        cls.logger.addHandler(sh)
        cls.logger.propagate = False

    @classmethod
    def set_logger_level(cls, log_level: str):
        level_dict = {'DEBUG': logging.DEBUG,
                      'INFO': logging.INFO,
                      'WARNING': logging.WARNING,
                      'ERROR': logging.ERROR,
                      'CRITICAL': logging.CRITICAL}

        cls.logger.handlers[-1].setLevel(level_dict[log_level.upper()])

    @classmethod
    def debug(cls, *args, **kwargs):
        if cls.logger is None:
            cls.init_logger(log_level=logging.DEBUG)
        return cls.logger.debug(*args, **kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        if cls.logger is None:
            cls.init_logger(log_level=logging.DEBUG)
        return cls.logger.info(*args, **kwargs)

    @classmethod
    def warn(cls, *args, **kwargs):
        if cls.logger is None:
            cls.init_logger(log_level=logging.DEBUG)
        return cls.logger.warning(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        if cls.logger is None:
            cls.init_logger(log_level=logging.DEBUG)
        return cls.logger.error(*args, **kwargs)

    @classmethod
    def critical(cls, *args, **kwargs):
        if cls.logger is None:
            cls.init_logger(log_level=logging.DEBUG)
        return cls.logger.critical(*args, **kwargs)


log = _Log


def get_timestamp(timestamp: Union[int, float] = None) -> str:
    datetime_ = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
    return datetime_.strftime('%y-%m-%d_%H-%M-%S')


class _NamespaceWrapper(object):
    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, item):
        try:
            value = self._data[item]
        except KeyError:
            value = getattr(self._data, item)
        if isinstance(value, dict):
            return _NamespaceWrapper(value)
        else:
            return value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return f'Wrapper({self._data})'


class YamlLoader(_NamespaceWrapper):
    def __init__(self, path):
        loader = yaml.SafeLoader
        # See https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(path, encoding='utf-8') as f:
            super().__init__(yaml.load(f, Loader=loader))

    @property
    def data(self):
        return self._data

    @property
    def args(self):
        return ()

    @property
    def kwargs(self):
        return {}


def print_return(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        print(ret)
        return ret

    return wrapper


class AutoInc(object):
    idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        return self.idx
