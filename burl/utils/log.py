import logging
from logging import FileHandler, StreamHandler
from logging.handlers import SocketHandler
import os

import numpy as np
import torch

from burl.utils import g_cfg

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"


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


def get_logger(name=__name__,
               log_level=logging.INFO,
               fmt='[%(asctime)s] %(message)s',
               datefmt='%b%d %H:%M:%S'):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # path = os.path.join(g_cfg.log_dir, 'log.txt')
    # if not os.path.exists(dir_ := os.path.dirname(path)):
    #     os.makedirs(dir_)
    # fh = FileHandler(path)
    # fh.setLevel(logging.DEBUG)
    # log.addHandler(fh)
    # soh = SocketHandler('127.0.0.1', 19996)
    soh = SocketHandler('10.12.120.120', 19996)
    soh.setFormatter(logging.Formatter())
    log.addHandler(soh)
    formatter = ColoredFormatter(fmt, datefmt)
    sh = StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    log.addHandler(sh)
    return log


np.set_printoptions(3, linewidth=1000, suppress=True)
torch.set_printoptions(linewidth=1000, profile='short')
logger = get_logger()
logger.DEBUG = logging.DEBUG
logger.INFO = logging.INFO
logger.WARNING = logging.WARNING
logger.ERROR = logging.ERROR
logger.CRITICAL = logging.CRITICAL


def set_logger_level(log_level):
    logger.handlers[-1].setLevel(log_level)


if __name__ == '__main__':
    set_logger_level(logger.INFO)
    logger.debug(torch.Tensor([1.0000001, 2, 3]))
    logger.info(123123)
    logger.warning(123123)
    logger.error(123123)
    logger.critical(123123)
