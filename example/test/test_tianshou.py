import numpy as np
import torch
from tianshou.env import ShmemVectorEnv
from tianshou.utils.net.continuous import ActorProb
from torch import nn

from example.loct.network import ActorNet
from qdpgym import sim
from qdpgym.tasks.loct import RandomCommanderHookV0, LocomotionV0


def test_actor():
    device = 'cuda'
    net_a = ActorNet(
        78, 133,
        activation=nn.Tanh,
        device=device,
    )
    actor = ActorProb(
        net_a,
        (12,),
        max_action=1.,
        unbounded=True,
        device=device,
    ).to(device)

    print(actor(torch.randn(size=(1, 211))))


def make_loct_env():
    torch.set_num_threads(1)
    robot = sim.Aliengo(500, 'actuator_net', noisy=False)
    task = LocomotionV0()
    task.add_hook(RandomCommanderHookV0())
    arena = sim.Plain()
    # for reward, weight in cfg['reward_cfg'].items():
    task.add_reward('UnifiedLinearReward', 0.1)
    env = sim.QuadrupedEnv(robot, arena, task)
    return env


def test_mp():
    envs = ShmemVectorEnv(
        [make_loct_env for _ in range(4)]
    )

    print(envs.reset())
    print(envs.step(np.zeros((4, 12))))
    print(envs.reset())
