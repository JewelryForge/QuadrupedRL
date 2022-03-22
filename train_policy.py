import sys

sys.path.append('.')
from burl.rl.runner import PolicyTrainer
from burl.utils import g_cfg, log_warn, init_logger, parse_args, get_timestamp
import wandb


def update_cfg_from_args():
    abbrs = {'num_iters': 'num_iterations',
             'rand_dyn': 'random_dynamics',
             'centralized': 'use_centralized_curriculum'}
    for name, value in parse_args():
        if name == 'mp_train':
            g_cfg.use_wandb = True
            g_cfg.use_mp = True
            g_cfg.rendering = False
            log_warn(f'wandb: on')
            log_warn(f'multi-process: on')
        elif name == 'cuda' or name == 'cpu':
            g_cfg.device = name
        elif name == 'on_plain':
            g_cfg.trn_type = 'plain'
            log_warn(f'terrain type: plain')
        else:
            if name in abbrs:
                name = abbrs[name]
            if not hasattr(g_cfg, name):
                raise RuntimeError(f"g_cfg has no attribute named '{name}'")
            setattr(g_cfg, name, value)
            value = getattr(g_cfg, name)
            if isinstance(value, bool):
                log_warn(f"{name}: {'on' if value else 'off'}")
            else:
                log_warn(f'{name}: {type(value).__name__} -> {value}')


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
    wandb.init(project='teacher-student', name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    if not g_cfg.log_dir:
        if g_cfg.use_wandb:
            g_cfg.log_dir = f'log/{get_timestamp(wandb.run.start_time)}#{wandb.run.name}@{wandb.run.id}'
        else:
            g_cfg.log_dir = f'log/{get_timestamp()}'
    wandb.config.update(g_cfg.__dict__)
    runner = PolicyTrainer(g_cfg.task_type)
    runner.learn()


if __name__ == '__main__':
    main()
