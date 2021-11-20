import sys

sys.path.append('.')
from burl.rl.train import TaskParam, OnPolicyRunner


def main():
    param = TaskParam()
    param.num_envs = 32
    runner = OnPolicyRunner(param)
    runner.learn(100000, True)


if __name__ == '__main__':
    main()
