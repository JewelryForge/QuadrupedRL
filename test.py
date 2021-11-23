import sys

sys.path.append('.')

import os

from burl.sim import QuadrupedSim, TGEnv, EnvContainer
from burl.utils import make_cls
from burl.rl.a2c import ActorCritic, Teacher, Critic
from burl.utils.config import TaskParam

import torch


def main(model_dir):
    print(f'load model {model_dir}')
    cfg = TaskParam()
    cfg.num_envs = 1
    make_robot = make_cls(QuadrupedSim, on_rack=False, make_sensors=[],
                          frequency=cfg.execution_frequency)
    make_env = make_cls(TGEnv, make_robot=make_robot, sim_frequency=cfg.sim_frequency,
                        action_frequency=cfg.action_frequency, use_gui=True)

    env = EnvContainer(cfg.num_envs, make_env)
    device = 'cuda'
    actor_critic = ActorCritic(Teacher(), Critic(), init_noise_std=0.).to(device)
    actor_critic.load_state_dict(torch.load(model_dir)['model_state_dict'])

    privileged_obs, obs = env.init_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(device), critic_obs.to(device)

    for _ in range(10000):
        actions = actor_critic.act(obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(device), critic_obs.to(device), rewards.to(
            device), dones.to(device)
        # self.alg.process_env_step(rewards, dones, infos)
    # state_dict =
    # print(state_dict)


if __name__ == '__main__':
    log_dir = 'log/' + sorted(os.listdir('log'))[-1]
    # recent_log = 4000
    recent_log = max(int(m.lstrip('model_').rstrip('.pt')) for m in os.listdir(log_dir) if m.startswith('model'))
    main(os.path.join(log_dir, f'model_{recent_log}.pt'))
    # print(sorted())
