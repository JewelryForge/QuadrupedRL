from itertools import chain

import torch

from burl.alg.ac import Actor, Critic
from burl.utils import g_cfg


class RolloutStorage(object):
    class Transition(object):
        def __init__(self):
            self.actor_obs = None
            self.critic_obs = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 device,
                 num_envs,
                 storage_len,
                 actor_obs_shape,
                 critic_obs_shape,
                 actions_shape,
                 rewards_shape=(1,)):
        # Core
        self.device = torch.device(device)
        self.actor_obs = torch.zeros(storage_len, num_envs, *actor_obs_shape, device=self.device)
        self.critic_obs = torch.zeros(storage_len, num_envs, *critic_obs_shape, device=self.device)
        self.actions = torch.zeros(storage_len, num_envs, *actions_shape, device=self.device)
        self.rewards = torch.zeros(storage_len, num_envs, *rewards_shape, device=self.device)
        self.dones = torch.zeros(storage_len, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(storage_len, num_envs, 1, device=self.device)
        self.values = torch.zeros(storage_len, num_envs, 1, device=self.device)
        self.returns = torch.zeros(storage_len, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(storage_len, num_envs, 1, device=self.device)
        self.mu = torch.zeros(storage_len, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(storage_len, num_envs, *actions_shape, device=self.device)

        self.storage_len = storage_len
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.storage_len:
            raise AssertionError("Rollout buffer overflow")
        self.actor_obs[self.step].copy_(transition.actor_obs)
        self.critic_obs[self.step].copy_(transition.critic_obs)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, discount_v, discount_adv):
        advantage = 0
        for step in reversed(range(self.storage_len)):
            if step == self.storage_len - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            mask = 1.0 - self.dones[step].float()
            # GAE: Generalized Advantage Estimator
            delta = self.rewards[step] + mask * discount_v * next_values - self.values[step]
            advantage = delta + mask * discount_v * discount_adv * advantage
            self.returns[step] = advantage + self.values[step]  # FIXME: WHY NOT ASSIGN DIRECTLY

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.storage_len
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        actor_obs = self.actor_obs.flatten(0, 1)
        critic_obs = self.critic_obs.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                actor_obs_batch = actor_obs[batch_idx]
                critic_obs_batch = critic_obs[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch,
                       returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch)


class PPO(object):
    Optim = torch.optim.AdamW

    # Criterion = torch.nn.SmoothL1Loss

    def __init__(self, actor: Actor, critic: Critic):
        self.device = torch.device(g_cfg.dev)
        self.learning_rate = g_cfg.learning_rate
        self.num_envs = g_cfg.num_envs
        self.storage_len = g_cfg.storage_len
        self.gamma = g_cfg.gamma
        self.lambda_gae = g_cfg.lambda_gae
        self.schedule = g_cfg.lr_scheduler
        self.num_mini_batches = g_cfg.num_mini_batches
        self.repeat_times = g_cfg.repeat_times
        self.clip_ratio = g_cfg.clip_ratio
        self.clip_value_loss = g_cfg.clip_value_loss
        self.value_loss_coef = g_cfg.value_loss_coef
        self.entropy_coef = g_cfg.entropy_coef
        self.max_grad_norm = g_cfg.max_grad_norm

        self.actor, self.critic = actor.to(self.device), critic.to(self.device)
        self.optim = self.Optim(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        # self.criterion = self.Criterion()
        if self.schedule == 'adaptive':
            self.adaptive = True
            self.desired_kl = g_cfg.desired_kl
        else:
            self.adaptive = False
            if self.schedule == 'linearLR':
                self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, 1.0, 0.5, 5000)

        self.transition = RolloutStorage.Transition()
        self.storage = RolloutStorage(self.device,
                                      self.num_envs, self.storage_len,
                                      (self.actor.input_dim,),
                                      (self.critic.input_dim,),
                                      (self.actor.output_dim,),
                                      (self.critic.output_dim,))

    def parameters(self):
        return chain(self.actor.parameters(), self.critic.parameters())

    def act(self, actor_obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor.get_action(actor_obs).detach()
        self.transition.values = self.critic(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, time_outs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.rewards += self.gamma * self.transition.values.squeeze() * time_outs

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        last_values = self.critic(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lambda_gae)

    def adaptively_modify_lr(self, sigma_batch, mu_batch, old_sigma_batch, old_mu_batch):
        with torch.inference_mode():
            kl = torch.sum((sigma_batch / old_sigma_batch + 1e-5).log() - 0.5 +
                           (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                           / (2.0 * sigma_batch.square()), axis=-1)
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            for param_group in self.optim.param_groups:
                param_group['lr'] = self.learning_rate

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.repeat_times)
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch) in generator:

            self.actor.get_action(obs_batch)
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            value_batch = self.critic(critic_obs_batch)
            entropy_batch = self.actor.entropy
            if self.adaptive:
                self.adaptively_modify_lr(self.actor.action_std, self.actor.action_mean,
                                          old_sigma_batch, old_mu_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = advantages_batch.squeeze() * ratio
            surrogate_clipped = advantages_batch.squeeze() * ratio.clamp(1.0 - self.clip_ratio,
                                                                         1.0 + self.clip_ratio)
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

            # Value function loss
            # if self.clip_value_loss:
            #     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_ratio,
            #                                                                                     self.clip_ratio)
            #     value_losses = (value_batch - returns_batch).pow(2)
            #     value_losses_clipped = (value_clipped - returns_batch).pow(2)
            #     value_loss = torch.max(value_losses, value_losses_clipped).mean()
            # else:

            value_loss = (returns_batch - value_batch).pow(2).mean() / (value_batch.std() + 1e-6)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optim.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            self.learning_rate = self.optim.param_groups[0]['lr']
        num_updates = self.repeat_times * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
