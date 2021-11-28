import sys

sys.path.append('.')
from burl.rl.train import OnPolicyRunner
from burl.utils import g_cfg


def main():
    g_cfg.num_steps_per_env = 256
    g_cfg.num_iterations = 10000
    g_cfg.rendering_enabled = False
    g_cfg.sleeping_enabled = False
    g_cfg.num_envs = 8
    runner = OnPolicyRunner()
    runner.learn()


if __name__ == '__main__':
    main()
