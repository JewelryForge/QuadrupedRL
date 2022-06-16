import argparse
from collections import defaultdict
from typing import Callable, List, Union, Optional, Any, Dict

import gym
import numpy as np
import torch
import wandb
from tianshou.data import Batch
from tianshou.env import ShmemVectorEnv, VectorEnvNormObs
from tianshou.utils import RunningMeanStd, BaseLogger


class NormObsWrapper(gym.Wrapper):
    """An observation normalization wrapper for vectorized environments.

    :param bool update_obs_rms: whether to update obs_rms. Default to True.
    :param float clip_obs: the maximum absolute value for observation. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        env: gym.Env,
        update_obs_rms: bool = False,
        clip_obs: float = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:

        super(NormObsWrapper, self).__init__(env)
        # initialize observation running mean/std
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()
        self.clip_max = clip_obs
        self.eps = epsilon

    def reset(
        self, *args, **kwargs
    ):
        obs = self.env.reset(*args, **kwargs)
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self._norm_obs(obs)

    def step(
        self,
        action: np.ndarray,
    ):
        obs, rew, done, info = self.env.step(action)
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self._norm_obs(obs), rew, done, info

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
            obs = np.clip(obs, -self.clip_max, self.clip_max)
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms


def make_parallel_env(
    make_env: Callable[[], gym.Env],
    seed: Union[int, None, List[int]],
    training_num: int,
    test_num: int,
    obs_norm: bool
):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """

    env = make_env()
    train_envs = ShmemVectorEnv(
        [make_env for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv(
        [make_env for _ in range(test_num)]
    ) if test_num else None
    train_envs.seed(seed)
    test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs


def init_actor_critic(actor, critic):
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)


class MyWandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[argparse.Namespace] = None,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            entity=entity,
            monitor_gym=True,
            config=config,  # type: ignore
            save_code=True,
        ) if not wandb.run else wandb.run

        self._reward_info = defaultdict(float)
        self._reward_counter = 0
        self._callbacks_train: List[Callable[[], dict]] = []
        self._callbacks_test: List[Callable[[], dict]] = []

    def collect_reward_info(self, **kwargs) -> Batch:
        if 'rew' in kwargs:
            for k, v in kwargs['info']['reward_info'].items():
                self._reward_info[k] += v.mean()
            self._reward_counter += 1
        return Batch()

    def write(self, step_type: str, step: int, data: Dict[str, Any]) -> None:
        wandb.log(data, step=step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], Any]] = None,
    ) -> None:
        if save_checkpoint_fn:
            save_checkpoint_fn(epoch, env_step, gradient_step)

    def add_callback(
        self, callback: Callable[[], dict],
        period='train'
    ) -> None:
        if period == 'train':
            self._callbacks_train.append(callback)
        elif period == 'test':
            self._callbacks_test.append(callback)
        elif period == 'both':
            self._callbacks_train.append(callback)
            self._callbacks_test.append(callback)
        else:
            raise ValueError(f'Unknown period {period}')

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"],
                }
                for k, v in self._reward_info.items():
                    log_data[f'train/reward_info/{k}'] = v / self._reward_counter
                self._reward_info.clear()
                self._reward_counter = 0
                for callback in self._callbacks_train:
                    log_data.update(callback())
                self.write("train/env_step", step, log_data)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:

        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
            }
            for callback in self._callbacks_test:
                log_data.update(callback())
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step
