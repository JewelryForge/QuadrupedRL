import sys

sys.path.append('.')
from burl.rl.runner import PolicyTrainer
from burl.utils import g_cfg, log_warn, init_logger, parse_args
import wandb


def update_cfg_from_args():
    for name, value in parse_args().items():
        if not hasattr(g_cfg, name):
            print(f"g_cfg has no attribute named '{name}'")
        setattr(g_cfg, name, value)
        value = getattr(g_cfg, name)
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
        g_cfg.schedule = 'fixed'
    wandb.init(project='teacher-student', config=g_cfg.__dict__, name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    log_warn(f'Training on {g_cfg.device}')
    runner = PolicyTrainer()
    runner.learn()


if __name__ == '__main__':
    main()
