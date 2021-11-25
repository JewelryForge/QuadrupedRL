import sys

sys.path.append('.')
from burl.utils import TaskParam
from burl.rl.train import OnPolicyRunner


def main():
    param = TaskParam()
    param.train_param.num_steps_per_env = 128   # change to 256
    param.render_param.rendering_enabled = False
    param.render_param.sleeping_enabled = False
    param.train_param.num_envs = 8
    runner = OnPolicyRunner(param)
    runner.learn(10000, True)


if __name__ == '__main__':
    main()
