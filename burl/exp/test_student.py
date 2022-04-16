import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from burl.exp import find_log, find_csc, student_log_dir as log_dir
from burl.utils import g_cfg, set_logger_level, init_logger
from burl.rl.runner import StudentPlayer, JoystickStudentPlayer as JoystickPlayer

if __name__ == '__main__':
    g_cfg.trn_type = 'plain'
    g_cfg.test_profile()
    g_cfg.driving_mode = True
    # g_cfg.slow_down_rendering()
    g_cfg.history_len = 123
    g_cfg.add_disturbance = False
    g_cfg.random_dynamics = True
    g_cfg.actuator_net = 'history'
    init_logger()
    set_logger_level('debug')
    remote = False
    joystick_control = True
    if remote:
        model_path = find_csc(run_name='', time_=None, epoch=None)
        # model_path = find_huzhou(run_name='2.8.0.21m2', time_=None, epoch=None)
    else:
        model_path = find_log(run_name='0.6', time_=None, epoch=None, log_dir=log_dir)
    if joystick_control and JoystickPlayer.is_available():
        player = JoystickPlayer(model_path)
    else:
        player = StudentPlayer(model_path, 'randCmd')
    player.play(False)
