import sys

sys.path.append('.')
from burl.rl.runner import OnPolicyRunner
from burl.utils import g_cfg
import wandb


def update_cfg_from_args():
    args = sys.argv[1:]
    assert len(args) % 2 == 0
    for i in range(len(args) // 2):
        name, value = args[2 * i], args[2 * i + 1]
        assert name.startswith('--')
        name = name.removeprefix('--')
        assert hasattr(g_cfg, name)
        setattr(g_cfg, name, value)


def main():
    update_cfg_from_args()
    wandb.init(project='teacher-student', config=g_cfg.__dict__, name=g_cfg.run_name)
    runner = OnPolicyRunner()
    runner.learn()


if __name__ == '__main__':
    main()
