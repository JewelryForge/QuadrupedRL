import logging
import sys

import torch
import numpy as np
import os

sys.path.append('.')

from burl.sim import A1, TGEnv, EnvContainer
from burl.utils import make_cls, g_cfg, g_dev, logger
from burl.alg.ac import ActorCritic, ActorTeacher, Critic


class Player:
    def __init__(self, model_dir):
        g_cfg.rendering_enabled = True
        g_cfg.sleeping_enabled = True
        make_robot = make_cls(A1)
        make_env = make_cls(TGEnv, make_robot=make_robot)

        self.env = EnvContainer(1, make_env)
        g_cfg.init_noise_std = 0.
        self.actor_critic = ActorCritic(ActorTeacher(), Critic()).to(g_dev)
        self.actor_critic.load_state_dict(torch.load(model_dir)['model_state_dict'])
        logger.info(f'load model {model_dir}')

    def play(self):
        privileged_obs, obs = self.env.init_observations()

        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(g_dev), critic_obs.to(g_dev)

        for _ in range(2000):
            actions = self.actor_critic.act(obs)
            obs, privileged_obs, rewards, done, info = self.env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, done = obs.to(g_dev), critic_obs.to(g_dev), rewards.to(g_dev), done.to(g_dev)
            # print(self.env._envs[0].getActionSmoothness())
        # obs_list = ExtendedObservation.l
        # print(np.mean(obs_list, axis=0), np.std(obs_list, axis=0))


def main(model_dir):
    player = Player(model_dir)
    player.play()


if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000, suppress=True)
    logging.basicConfig(level=logging.DEBUG)
    log_dir = 'wandb/latest-run/files'
    recent_log = 2000
    # recent_log = max(int(m.removeprefix('model_').removesuffix('.pt'))
    #                  for m in os.listdir(log_dir) if m.startswith('model'))
    main(os.path.join(log_dir, f'model_{recent_log}.pt'))
    # print(sorted())
