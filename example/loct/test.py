#!/usr/bin/env python3

import argparse

import numpy as np
import torch
import yaml
from tianshou.data import Collector
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch import nn
from torch.distributions import Independent, Normal

from example.loct.network import ActorNet
from example.utils import NormObsWrapper
from qdpgym import sim
from qdpgym.tasks.loct import LocomotionV0, GamepadCommanderHook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    return parser.parse_args()


def make_loct_env(cfg):
    torch.set_num_threads(1)
    robot = sim.Aliengo(500, 'actuator_net', noisy=True)
    task = LocomotionV0()
    # task = LocomotionPMTG()

    if cfg['terrain'] == 'random':
        arena = sim.NullTerrain()
        task.add_hook(sim.RandomTerrainHook())
    elif cfg['terrain'] == 'plain':
        arena = sim.Plain()
    else:
        raise ValueError(f'Unknown terrain `{cfg.terrain}`')
    if cfg['perturb']:
        task.add_hook(sim.RandomPerturbHook())

    task.add_hook(GamepadCommanderHook())
    # task.add_hook(sim.HeightSampleVisualizer())
    # task.add_hook(RandomCommanderHookV0())
    # task.add_hook(sim.VideoRecorderHook())
    task.add_hook(sim.ExtraViewerHook())
    for reward, weight in cfg['reward_cfg'].items():
        task.add_reward(reward, weight)

    if 'reward_coeff' in cfg:
        task.set_reward_coeff(cfg['reward_coeff'])
    env = sim.QuadrupedEnv(robot, arena, task)
    return env


def test_ppo(args):
    with open(args.task, encoding='utf-8') as f:
        task_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    def make_env():
        return make_loct_env(task_cfg)

    obs_norm = False
    env = make_env()
    if obs_norm:
        env = NormObsWrapper(env)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = ActorNet(78, 132, device=args.device)
    actor = ActorProb(
        net_a,
        env.action_space.shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        env.observation_space.shape,
        hidden_sizes=(256, 256, 256),
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters())
    )

    policy = PPOPolicy(
        actor, critic, optim,
        lambda *logits: Independent(Normal(*logits), 1)
    )
    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        if obs_norm:
            env.set_obs_rms(ckpt['obs_rms'])
        print("Loaded agent from: ", args.resume_path)
    test_collector = Collector(policy, env)

    # Let's watch its performance!
    policy.eval()
    test_collector.reset()
    result = test_collector.collect(n_episode=100, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    args = parse_args()
    test_ppo(args)
