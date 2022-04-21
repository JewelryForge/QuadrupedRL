import os
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from burl.exp import teacher_log_dir as log_dir, get_timestamp, update_cfg_from_args
from burl.rl.runner import PolicyTrainer
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
    log_warn(f'Training on {g_cfg.device}')
    wandb.init(project='policy-fine-tune', name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    if not g_cfg.log_dir:
        if g_cfg.use_wandb:
            start_time, run_name, run_id = wandb.run.start_time, wandb.run.name, wandb.run.id
            g_cfg.log_dir = os.path.join(log_dir, f'{get_timestamp(start_time)}#{run_name}@{run_id}')
        else:
            g_cfg.log_dir = os.path.join(log_dir, f'{get_timestamp()}')
    wandb.config.update(g_cfg.__dict__)
    runner = PolicyTrainer(g_cfg.task_type)
    runner.learn()


if __name__ == '__main__':
    main()
