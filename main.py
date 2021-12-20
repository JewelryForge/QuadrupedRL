import sys

sys.path.append('.')
from burl.rl.runner import OnPolicyRunner
from burl.rl.task import BasicTask
from burl.utils import g_cfg, log_warn, init_logger
import wandb


def update_cfg_from_args():
    args = iter(sys.argv[1:])
    while True:
        try:
            name = next(args)
        except StopIteration:
            break
        assert name.startswith('--')
        name = name.removeprefix('--')
        if '=' in name:
            name, value = name.split('=')
        else:
            try:
                value = next(args)
                assert not value.startswith('--')
            except (StopIteration, AssertionError):
                raise RuntimeError(f"Parameter named '{name}' has no corresponding value")
        name = name.replace('-', '_')
        if not hasattr(g_cfg, name):
            print(f"g_cfg has no attribute named '{name}'")
        setattr(g_cfg, name, value)
        value = getattr(g_cfg, name)
        log_warn(f'{name}: {type(value).__name__} -> {value}')


def main():
    init_logger()
    g_cfg.task_class = BasicTask
    g_cfg.num_envs = 1
    g_cfg.trn_type = 'plain'
    g_cfg.rendering = True
    g_cfg.use_mp = False
    g_cfg.use_wandb = False
    g_cfg.sleeping_enabled = False
    g_cfg.schedule = 'fixed'
    update_cfg_from_args()
    # g_cfg._rewards_weights = [(r.__class__.__name__, w) for r, w in BasicTask.rewards_weights]
    wandb.init(project='teacher-student', config=g_cfg.__dict__, name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    log_warn(f'Training on {g_cfg.device}')
    runner = OnPolicyRunner()
    runner.learn()


if __name__ == '__main__':
    main()
