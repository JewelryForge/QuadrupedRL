import logging
import os
from logging.handlers import SocketHandler
from typing import Optional

import numpy as np
import torch

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


logger: Optional[logging.Logger] = None


def init_logger(name=__name__,
                log_level=logging.INFO,
                log_fmt='[%(asctime)s] %(message)s',
                date_fmt='%b%d %H:%M:%S',
                log_dir=None,
                client_ip='127.0.0.1'):
    np.set_printoptions(3, linewidth=1000, suppress=True)
    torch.set_printoptions(linewidth=1000, profile='short')
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_dir:
        os.makedirs(log_dir)
        fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    soh = SocketHandler(client_ip, 19996)
    # soh = SocketHandler('10.12.120.120', 19996)
    soh.setFormatter(logging.Formatter())
    logger.addHandler(soh)
    formatter = ColoredFormatter(log_fmt, date_fmt)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def set_logger_level(log_level: str):
    level_dict = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}

    logger.handlers[-1].setLevel(level_dict[log_level.upper()])


def log_debug(*args, **kwargs):
    return logger.debug(*args, **kwargs)


def log_info(*args, **kwargs):
    return logger.info(*args, **kwargs)


def log_warn(*args, **kwargs):
    return logger.warning(*args, **kwargs)


def log_error(*args, **kwargs):
    return logger.error(*args, **kwargs)


def log_critical(*args, **kwargs):
    return logger.critical(*args, **kwargs)


if __name__ == '__main__':
    init_logger()
    set_logger_level('DEBUG')
    logger.debug(torch.Tensor([1.0000001, 2, 3]))
    logger.info(123123)
    logger.warning(123123)
    logger.error(123123)
    logger.critical(123123)
