import sys

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

        for _ in range(10000):
            actions = self.actor_critic.act(obs)
            obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                self.device), dones.to(self.device)


def main(model_dir):
    player = Player(model_dir)
    player.play()


if __name__ == '__main__':
    log_dir = 'log/' + sorted(os.listdir('log'))[-1]
    # recent_log = 1400
    recent_log = max(int(m.lstrip('model_').rstrip('.pt')) for m in os.listdir(log_dir) if m.startswith('model'))
    main(os.path.join(log_dir, f'model_{recent_log}.pt'))
    # print(sorted())
