import logging

import numpy as np
import torch

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
    formatter = ColoredFormatter(fmt, datefmt)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger(name)
    log.setLevel(log_level)
    log.addHandler(stream)
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
    logger.setLevel(log_level)
    for h in logger.handlers:
        h.setLevel(log_level)


if __name__ == '__main__':
    set_logger_level(logger.DEBUG)
    logger.debug(torch.Tensor([1.0000001, 2, 3]))
    logger.info(123123)
    logger.warning(123123)
    logger.error(123123)
    logger.critical(123123)
