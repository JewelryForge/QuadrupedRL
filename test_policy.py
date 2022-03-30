import sys

sys.path.append('.')

from burl.utils import g_cfg, set_logger_level, init_logger, find_log, find_log_remote, make_part
from burl.rl.runner import TeacherPlayer, JoystickTeacherPlayer as JoystickPlayer

find_csc = make_part(find_log_remote, host='csc')
find_huzhou = make_part(find_log_remote, host='huzhou')

if __name__ == '__main__':
    g_cfg.trn_type = 'curriculum'
    g_cfg.test_profile()
    g_cfg.driving_mode = True
    # g_cfg.slow_down_rendering()
    g_cfg.add_disturbance = True
    g_cfg.random_dynamics = True
    g_cfg.actuator_net = 'history'
    # g_cfg.latency_range = (0.02, 0.03)
    init_logger()
    set_logger_level('debug')
    remote = False
    if remote:
        model_path = find_csc(run_name='', time_=None, epoch=None)
        # model_path = find_huzhou(run_name='2.8.0.21m2', time_=None, epoch=None)
    else:
        model_path = find_log(run_name='2.8.0.22m2', time_=None, epoch=None)
    if JoystickPlayer.is_available():
        player = JoystickPlayer(model_path)
    else:
        player = TeacherPlayer(model_path, 'randCmd')
    player.play(True)
