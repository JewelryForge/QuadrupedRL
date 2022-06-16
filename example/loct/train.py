import pybullet
import argparse
import os
import pprint

import numpy as np
import torch
import wandb
import yaml
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from example.loct.network import ActorNet
from example.utils import init_actor_critic, MyWandbLogger
from qdpgym import sim
from qdpgym.tasks.loct import RandomCommanderHookV0, RandomCommanderHookV05, RandomCommanderHookV1, \
    LocomotionV0, CommandRewardAnalyser
from qdpgym.utils import get_timestamp, AutoInc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=8192)
    parser.add_argument("--repeat-per-collect", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--obs-norm", type=int, default=0)

    # ppo special
    parser.add_argument("--rew-norm", type=int, default=1)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="tanh")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="loct")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.task, encoding='utf-8') as f:
        task_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cmd_rew_analyser = CommandRewardAnalyser(200)


    def make_loct_env(cfg, train=True):
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
        if train:
            task.add_hook(
                cmd_rew_analyser('RotationReward'),
                'CommandRewardAnalyser'
            )
        task.add_hook(RandomCommanderHookV05())
        for reward, weight in cfg['reward_cfg'].items():
            task.add_reward(reward, weight)

        if 'reward_coeff' in cfg:
            task.set_reward_coeff(cfg['reward_coeff'])
        env = sim.QuadrupedEnv(robot, arena, task)
        return env


    env = make_loct_env(task_cfg)
    train_envs = ShmemVectorEnv([
        lambda: make_loct_env(task_cfg)
        for _ in range(args.training_num)
    ])
    test_envs = ShmemVectorEnv([
        lambda: make_loct_env(task_cfg, False)
        for _ in range(args.test_num)
    ]) if args.test_num else None

    if args.obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())

    obs_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", obs_shape)
    print("Actions shape:", action_shape)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = ActorNet(78, 133, device=args.device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=1.0,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        obs_shape,
        hidden_sizes=(256, 256, 256),
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -2.3)
    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr
    )
    init_actor_critic(actor, critic)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )


    def dist(*logits):
        return Independent(Normal(*logits), 1)


    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        if args.obs_norm:
            train_envs.set_obs_rms(ckpt["obs_rms"])
            test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # logger
    logger = MyWandbLogger(
        project=args.wandb_project,
        name=args.run_name,
        run_id=args.resume_id,
        config=args,
    )
    start_time, run_name, run_id = wandb.run.start_time, wandb.run.name, wandb.run.id
    log_path = os.path.join(args.logdir, f'{get_timestamp(start_time)}#{run_name}@{run_id}')
    os.makedirs(log_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.step_per_collect, len(train_envs))
    else:
        buffer = ReplayBuffer(args.step_per_collect)

    train_collector = Collector(
        policy, train_envs, buffer,
        preprocess_fn=logger.collect_reward_info,
        exploration_noise=True
    )
    test_collector = Collector(
        policy, test_envs
    ) if test_envs is not None else None


    def analyse_callback():
        fig1, fig2 = cmd_rew_analyser.analyse()
        return {
            'curricula/reward': wandb.Image(fig1, mode='L'),
            'curricula/weight': wandb.Image(fig2, mode='L'),
        }


    logger.add_callback(
        analyse_callback, 'test'
    )


    def save_best_fn(policy):
        state = {"model": policy.state_dict()}
        if args.obs_norm:
            state["obs_rms"] = train_envs.get_obs_rms()
        torch.save(state, os.path.join(log_path, "policy.pth"))


    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int):
        state = {"model": policy.state_dict()}
        if args.obs_norm:
            state["obs_rms"] = train_envs.get_obs_rms()
        path = os.path.join(log_path, f"policy_{epoch}.pth")
        torch.save(state, path)
        return path


    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,
    )
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
