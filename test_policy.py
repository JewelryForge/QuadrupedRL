import sys

sys.path.append('.')

from burl.utils import g_cfg, set_logger_level, init_logger, find_log, find_log_remote
from burl.rl.runner import PolicyPlayer

if __name__ == '__main__':
    g_cfg.trn_type = 'plain'
    g_cfg.trn_roughness = 0.05
    g_cfg.sleeping_enabled = True
    g_cfg.on_rack = False
    g_cfg.test_mode = True
    g_cfg.rendering = True
    g_cfg.single_step_rendering = False
    g_cfg.add_disturbance = True
    g_cfg.tg_init = 'symmetric'
    init_logger()
    set_logger_level('debug')
    remote = True
    if remote:
        # model = find_log_remote(time=None, epoch=None, log_dir='teacher-student/log')
        model_path = find_log_remote(time=None, epoch=None, log_dir='Workspaces/teacher-student/log',
                                     host='jewel@10.12.120.120', port=22)
        # model = find_log_remote(time=None, epoch=None, log_dir='python_ws/ts-dev/log',
        #                         host='jewelry@10.192.119.171', port=22)
    else:
        model_path = find_log(time=None, epoch=None)
    player = PolicyPlayer(model_path)
    player.play()
