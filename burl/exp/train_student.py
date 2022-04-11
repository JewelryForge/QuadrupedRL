import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from burl.exp import student_log_dir as log_dir, update_cfg_from_args, find_log, get_timestamp
from burl.rl.imitate import ImitationRunner
from burl.utils import g_cfg, log_warn, init_logger
import wandb


def main():
    init_logger()
    if len(sys.argv) > 1:
        update_cfg_from_args()
    else:
        g_cfg.num_envs = 1
        g_cfg.trn_type = 'plain'
        g_cfg.rendering = True
        g_cfg.use_mp = False
        g_cfg.use_wandb = False
        g_cfg.sleeping_enabled = False
        g_cfg.lr_scheduler = 'fixed'
        g_cfg.model_path = find_log()
    log_warn(f'Training on {g_cfg.device}')

    if not os.path.exists(g_cfg.model_path):
        if g_cfg.model_path.startswith('find'):
            args = [arg_s for arg in g_cfg.model_path.removeprefix('find(').removesuffix(')').split(',')
                    if (arg_s := arg.strip())]
            arg_len = len(args)
            if arg_len == 0:
                g_cfg.model_path = find_log()
            elif arg_len == 1:
                g_cfg.model_path = find_log(run_name=args[0])
            elif arg_len == 2:
                g_cfg.model_path = find_log(run_name=args[0], epoch=int(args[1]))
            else:
                raise RuntimeError(f'Unknown find_log args')
        else:
            raise RuntimeError(f'`{g_cfg.model_path} not exists')

    wandb.init(project='imitate', name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    if not g_cfg.log_dir:
        if g_cfg.use_wandb:
            start_time, run_name, run_id = wandb.run.start_time, wandb.run.name, wandb.run.id
            g_cfg.log_dir = os.path.join(log_dir, f'{get_timestamp(start_time)}#{run_name}@{run_id}')
        else:
            g_cfg.log_dir = os.path.join(log_dir, f'{get_timestamp()}')
    wandb.config.update(g_cfg.__dict__)
    runner = ImitationRunner(g_cfg.model_path, g_cfg.task_type)
    runner.learn()


if __name__ == '__main__':
    main()
