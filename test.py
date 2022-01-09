import sys

sys.path.append('.')

import torch

from burl.sim import A1, AlienGo, TGEnv, EnvContainer
from burl.utils import make_cls, g_cfg, log_info, set_logger_level, to_dev, init_logger, find_log, find_log_remote
from burl.alg.ac import ActorCritic, Actor, Critic
from burl.rl.state import ExteroObservation, ProprioObservation, Action, ExtendedObservation


class Player:
    def __init__(self, model_dir, make_robot=A1):
        make_env = make_cls(TGEnv, make_robot=make_robot)
        self.env = EnvContainer(make_env, 1)
        self.actor_critic = ActorCritic(
            Actor(ExteroObservation.dim, ProprioObservation.dim, Action.dim,
                  g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims),
            Critic(ExtendedObservation.dim, 1), g_cfg.init_noise_std).to(g_cfg.dev)
        log_info(f'Loading model {model_dir}')
        self.actor_critic.load_state_dict(torch.load(model_dir)['model_state_dict'])

    def play(self):
        p_obs, obs = to_dev(*self.env.init_observations())

        for _ in range(20000):
            actions = self.actor_critic.act_inference(obs)
            p_obs, obs, _, dones, info = self.env.step(actions)
            p_obs, obs = to_dev(p_obs, obs)

            if any(dones):
                print(self.env._envs[0].robot._sum_work)
                reset_ids = torch.nonzero(dones)
                p_obs_reset, obs_reset = to_dev(*self.env.reset(reset_ids))
                p_obs[reset_ids,], obs[reset_ids,] = p_obs_reset, obs_reset
                print(info['episode_reward'])
            # time.sleep(0.05)


def main(model_dir):
    player = Player(model_dir, AlienGo)
    player.play()


if __name__ == '__main__':
    g_cfg.trn_type = 'plain'
    g_cfg.trn_roughness = 0.05
    g_cfg.sleeping_enabled = True
    g_cfg.on_rack = False
    g_cfg.test_mode = True
    g_cfg.rendering = True
    g_cfg.single_step_rendering = False
    g_cfg.add_disturbance = True
    g_cfg.tg_init = 'symmetric'
    init_logger()
    set_logger_level('debug')
    remote = False
    if remote:
        model = find_log_remote(time=None, epoch=None, log_dir='teacher-student/log')
        # model = find_log_remote(time=None, epoch=None, log_dir='python_ws/ts-dev/log',
        #                         host='jewelry@10.192.119.171', port=22)
    else:
        model = find_log(time=None, epoch=None)
    main(model)
