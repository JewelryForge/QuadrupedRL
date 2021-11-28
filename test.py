import logging
import sys
import time

import numpy as np

sys.path.append('.')

import os

from burl.sim import A1, TGEnv, EnvContainer
from burl.utils import make_cls
from burl.alg.ac import ActorCritic, ActorTeacher, Critic
from burl.utils.config import TaskParam

import torch


class Player:
    def __init__(self, model_dir, param=TaskParam(), device='cuda'):
        self.cfg = param.train_param
        param.render_param.rendering_enabled = True
        # param.render_param.sleeping_enabled = True
        make_robot = make_cls(A1, physics_param=param.sim_param)
        make_env = make_cls(TGEnv, make_robot=make_robot,
                            sim_param=param.sim_param, render_param=param.render_param)

        self.env = EnvContainer(1, make_env)

        self.device = torch.device(device)
        self.actor_critic = ActorCritic(ActorTeacher(), Critic(), init_noise_std=0.).to(self.device)
        self.actor_critic.load_state_dict(torch.load(model_dir)['model_state_dict'])
        print(f'load model {model_dir}')

    def play(self):
        privileged_obs, obs = self.env.init_observations()

        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        for _ in range(2000):
            actions = self.actor_critic.act(obs)
            obs, privileged_obs, rewards, done, info = self.env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, done = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                self.device), done.to(self.device)
            # print(self.env._envs[0].getActionSmoothness())
        # obs_list = ExtendedObservation.l
        # print(np.mean(obs_list, axis=0), np.std(obs_list, axis=0))


def main(model_dir):
    player = Player(model_dir)
    player.play()


if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000, suppress=True)
    logging.basicConfig(level=logging.DEBUG)
    log_dir = 'log/' + sorted(os.listdir('log'))[-1]
    # recent_log = 2000
    recent_log = max(int(m.removeprefix('model_').removesuffix('.pt'))
                     for m in os.listdir(log_dir) if m.startswith('model'))
    main(os.path.join(log_dir, f'model_{recent_log}.pt'))
    # print(sorted())
