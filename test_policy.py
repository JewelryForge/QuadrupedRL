import sys

sys.path.append('.')

from burl.utils import g_cfg, set_logger_level, init_logger, find_log, find_log_remote
from burl.rl.runner import PolicyPlayer

if __name__ == '__main__':
    g_cfg.trn_type = 'plain'
    g_cfg.trn_roughness = 0.05
    g_cfg.test_profile()
    # g_cfg.slow_down_rendering()
    g_cfg.add_disturbance = True
    g_cfg.actuator_net = False
    g_cfg.tg_init = 'symmetric'
    init_logger()
    set_logger_level('debug')
    remote = False
    if remote:
        model_path = find_log_remote(time=None, epoch=None, log_dir='teacher-student/log')
        # model_path = find_log_remote(time=None, epoch=None, log_dir='python_ws/ts-dev/log',
        #                         host='jewelry@10.192.119.171', port=22)
    else:
        model_path = find_log(time=None, epoch=None)
    player = PolicyPlayer(model_path)
    # player = PolicyPlayer('/data/teacher-student-backup/log/Feb15_13-59-29/model_2200.pt')
    player.play()
