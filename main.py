import sys

sys.path.append('.')
from burl.rl.runner import OnPolicyRunner
from burl.rl.task import BasicTask
from burl.utils import g_cfg, logger
import wandb


def update_cfg_from_args():
    args = sys.argv[1:]
    assert len(args) % 2 == 0
    for i in range(len(args) // 2):
        name, value = args[2 * i], args[2 * i + 1]
        assert name.startswith('--')
        name = name.removeprefix('--')
        name = name.replace('-', '_')
        assert hasattr(g_cfg, name)
        setattr(g_cfg, name, value)
        value = getattr(g_cfg, name)
        logger.warning(f'{name}: {type(value).__name__} -> {value}')


def main():
    update_cfg_from_args()
    g_cfg.task_class = BasicTask
    # g_cfg.num_envs = 1
    # g_cfg.use_trn_curriculum = True
    # g_cfg.rendering = True
    # g_cfg.use_mp = False
    # g_cfg.use_wandb = False
    # g_cfg.sleeping_enabled = False
    g_cfg.rewards_weights = [(r.__class__.__name__, w) for r, w in BasicTask.rewards]
    wandb.init(project='teacher-student', config=g_cfg.__dict__, name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')

    runner = OnPolicyRunner()
    runner.learn()


if __name__ == '__main__':
    main()
