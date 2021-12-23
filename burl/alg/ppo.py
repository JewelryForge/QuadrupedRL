import torch

from burl.alg.ac import ActorCritic
from burl.utils import g_cfg


class RolloutStorage(object):
    class Transition(object):
        def __init__(self):
            self.observations = None
            self.critic_observations = None
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

    def __init__(self, num_envs, num_transitions_per_env,
                 obs_shape, privileged_obs_shape, actions_shape):
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        g_dev = g_cfg.dev
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=g_dev)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs,
                                                       *privileged_obs_shape, device=g_dev)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=g_dev)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=g_dev)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=g_dev)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=g_dev)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
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
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
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
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=g_cfg.dev)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

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

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                      old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None


class PPO(object):
    actor_critic: ActorCritic

    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.actor_critic.to(g_cfg.dev)
        self.optimizer = torch.optim.AdamW(self.actor_critic.parameters(), lr=g_cfg.learning_rate,
                                           weight_decay=1e-2)
        if g_cfg.schedule == 'linearLR':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5,
                                                               total_iters=2000)
        self.transition = RolloutStorage.Transition()
        self.storage = RolloutStorage(g_cfg.num_envs, g_cfg.storage_len, (g_cfg.p_obs_dim,), (g_cfg.p_obs_dim,),
                                      (g_cfg.action_dim,))
        self.learning_rate = g_cfg.learning_rate

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, time_outs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.rewards += g_cfg.gamma * torch.squeeze(
            self.transition.values * time_outs.unsqueeze(1).to(g_cfg.dev), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        # self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, g_cfg.gamma, g_cfg.lambda_)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        cfg = g_cfg
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(cfg.num_mini_batches, cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(cfg.num_mini_batches, cfg.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if cfg.desired_kl is not None and cfg.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > cfg.desired_kl * 2.0:
                        learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < cfg.desired_kl / 2.0:
                        learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - cfg.clip_param,
                                                                               1.0 + cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-cfg.clip_param,
                                                                                                cfg.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), cfg.max_grad_norm)
            self.learning_rate = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        num_updates = cfg.num_learning_epochs * cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
