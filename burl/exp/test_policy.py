import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from burl.exp import find_log, find_csc, find_wuzhen
from burl.utils import g_cfg
from burl.rl.runner import TeacherPlayer, JoystickTeacherPlayer as JoystickPlayer

if __name__ == '__main__':
    g_cfg.trn_type = 'curriculum'
    g_cfg.test_profile()
    g_cfg.driving_mode = True
    # g_cfg.slow_down_rendering()
    g_cfg.add_disturbance = True
    g_cfg.random_dynamics = True
    g_cfg.actuator_net = 'history'
    # g_cfg.latency_range = (0.02, 0.03)
    remote = True
    joystick_control = True
    if remote:
        model_path = find_wuzhen(run_name='1.3.0NoEnd', run_id='', time_=None, epoch=None)
        # model_path = find_huzhou(run_name='2.8.0.21m2', time_=None, epoch=None)
    else:
        model_path = find_log(run_name='1.2.1', run_id='', time_=None, epoch=None)
    if joystick_control and JoystickPlayer.is_available():
        player = JoystickPlayer(model_path)
    else:
        player = TeacherPlayer(model_path, 'randCmd', seed=2)
    player.play(allow_reset=False)
