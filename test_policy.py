import sys

sys.path.append('.')

from burl.utils import g_cfg, set_logger_level, init_logger, find_log, find_log_remote
from burl.rl.runner import PolicyPlayer, JoystickPlayer

if __name__ == '__main__':
    g_cfg.trn_type = 'curriculum'
    g_cfg.test_profile()
    # g_cfg.slow_down_rendering()
    g_cfg.add_disturbance = True
    g_cfg.random_dynamics = True
    g_cfg.actuator_net = 'history'
    g_cfg.tg_init = 'symmetric'
    init_logger()
    set_logger_level('debug')
    remote = False
    if remote:
        model_path = find_log_remote(host='csc', time_=None, epoch=None, log_dir='teacher-student/log')
    else:
        model_path = find_log(time_=None, epoch=None)
    if JoystickPlayer.is_available():
        player = JoystickPlayer(model_path)
    else:
        player = PolicyPlayer(model_path, 'randCmd')
    player.play()
