import sys

import torch
import os

sys.path.append('.')

from burl.sim import A1, TGEnv, EnvContainer
from burl.utils import make_cls, g_cfg, logger, set_logger_level, str2time, to_dev
from burl.alg.ac import ActorCritic, ActorTeacher, Critic


class Player:
    def __init__(self, model_dir):
        g_cfg.rendering = True
        g_cfg.sleeping_enabled = True
        make_robot = A1
        make_env = make_cls(TGEnv, make_robot=make_robot)

        self.env = EnvContainer(make_env, 1)
        g_cfg.init_noise_std = 0.
        self.actor_critic = ActorCritic(ActorTeacher(), Critic()).to(g_cfg.dev)
        self.actor_critic.load_state_dict(torch.load(model_dir)['model_state_dict'])
        logger.info(f'load model {model_dir}')

    def play(self):
        privileged_obs, obs = self.env.init_observations()

        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = to_dev(obs, critic_obs)

        for _ in range(2000):
            actions = self.actor_critic.act(obs)
            obs, privileged_obs, *_ = self.env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs = to_dev(obs, critic_obs)
            # print(self.env._envs[0].getActionSmoothness())
        # obs_list = ExtendedObservation.l
        # print(np.mean(obs_list, axis=0), np.std(obs_list, axis=0))


def main(model_dir):
    player = Player(model_dir)
    player.play()


def find_log(time=None, epoch=None):
    local_logs = [l for l in os.listdir('log') if not l.startswith('remote')]
    folders = sorted(local_logs, key=str2time, reverse=True)
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


def find_log_remote(host='61.153.52.71', port=10022, log_dir='teacher-student/log', time=None, epoch=None):
    remote_logs = os.popen(f'ssh {host} -p {port} ls {log_dir}').read().split('\n')
    remote_logs.remove('')
    folders = sorted(remote_logs, key=str2time, reverse=True)
    if not time:
        folder = folders[0]
    else:
        for f in folders:
            if ''.join(f.split('_')[1].split('-')).startswith(time):
                folder = f
                break
        else:
            raise RuntimeError(f'Record with time {time} not found, all {folders}')
    models = os.popen(f'ssh {host} -p {port} ls {os.path.join(log_dir, folder)}').read().split('\n')
    final_epoch = max(int(m.removeprefix('model_').removesuffix('.pt'))
                      for m in models if m.startswith('model'))
    if epoch:
        if epoch > final_epoch:
            raise RuntimeError(f'Epoch {epoch} does not exist, max {final_epoch}')
    else:
        epoch = final_epoch
    model_name = f'model_{epoch}.pt'
    remote_log = os.path.join(log_dir, folder, model_name)
    local_log_dir = os.path.join('log', 'remote-' + folder)
    os.makedirs(local_log_dir, exist_ok=True)
    flag = os.system(f'scp -P {port} {host}:{remote_log} {local_log_dir}')
    if flag:
        raise RuntimeError('scp failed')
    return os.path.join(local_log_dir, model_name)


if __name__ == '__main__':
    # g_cfg.plain = True
    g_cfg.trn_roughness = 0.0
    set_logger_level(logger.DEBUG)
    remote = True
    time = None
    epoch = None
    if remote:
        model = find_log_remote(time=time, epoch=epoch)
    else:
        model = find_log(time=time, epoch=epoch)
    main(model)
