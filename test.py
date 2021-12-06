import sys

import torch
import numpy as np
import os

sys.path.append('.')

from burl.sim import A1, TGEnv, EnvContainer
from burl.utils import make_cls, g_cfg, g_dev, logger, set_logger_level, str2time
from burl.alg.ac import ActorCritic, ActorTeacher, Critic


class Player:
    def __init__(self, model_dir):
        g_cfg.rendering = True
        g_cfg.sleeping_enabled = True
        make_robot = A1
        make_env = make_cls(TGEnv, make_robot=make_robot)

        self.env = EnvContainer(make_env, 1)
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


def find_log(time=None, epoch=None):
    folders = sorted(os.listdir('log'), key=str2time, reverse=True)
    if not time:
        folder = folders[0]
    else:
        for f in folders:
            if ''.join(f.split('_')[1].split('-')).startswith(time):
                folder = f
                break
        else:
            raise RuntimeError(f'Record with time {time} not found')
    folder = os.path.join('log', folder)
    final_epoch = max(int(m.removeprefix('model_').removesuffix('.pt'))
                      for m in os.listdir(folder) if m.startswith('model'))
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    return os.path.join(folder, f'model_{epoch}.pt')


if __name__ == '__main__':
    g_cfg.plain = True
    # g_cfg.trn_roughness = 0.03
    set_logger_level(logger.DEBUG)
    main(find_log(time='1629', epoch=6800))
