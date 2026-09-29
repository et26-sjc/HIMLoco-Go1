"""Stage 1 PPO-Lagrangian with separate cost GAE and intact HIM updates."""

import torch
from torch import nn

from .quietpaw_him_ppo import QuietPawHIMPPO
from .quietpaw_pid import QuietPawPID
from .quietpaw_assignment import balanced_assignment
from rsl_rl.storage.quietpaw_cmdp_storage import QuietPawCMDPStorage


class QuietPawConstrainedPPO(QuietPawHIMPPO):
    def __init__(self, *args, cost_value_loss_coef=1.0, pid=None, epsilon_seed=1, epsilon_bins=8, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.gamma < 1:
            raise ValueError("Normalized discounted cost requires 0 < gamma < 1")
        self.cost_value_loss_coef = cost_value_loss_coef
        self.pid_cfg = pid or {}
        self.iterations = 0
        self.diagnostics = {}
        self.epsilon_seed, self.epsilon_bins = epsilon_seed, epsilon_bins

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        if num_envs < 2:
            raise ValueError("Stage 1 training needs >=2 environments for the epsilon grid.")
        self.storage = QuietPawCMDPStorage(num_envs, num_transitions_per_env,
                                         actor_obs_shape, critic_obs_shape, action_shape, self.device)
        self.transition = QuietPawCMDPStorage.Transition()
        # Single-cohort initialization for standalone algorithm tests. The runner
        # replaces this with the actual joint terrain/command stratification.
        self.epsilon, self.epsilon_labels = balanced_assignment(
            torch.zeros(num_envs), torch.zeros(num_envs), self.epsilon_seed,
            min(self.epsilon_bins, num_envs))
        self.epsilon = self.epsilon.to(self.device)
        self.epsilon_labels = self.epsilon_labels.to(self.device)
        self.pid = QuietPawPID(self.epsilon, **self.pid_cfg)
        self.episode_cost = torch.zeros_like(self.epsilon)
        self.episode_discount = torch.ones_like(self.epsilon)
        self.completed_cost = torch.zeros_like(self.epsilon)
        self.completed_count = torch.zeros_like(self.epsilon)
        self.metric_sums = {}
        self.metric_steps = 0
        self.success_count = torch.zeros_like(self.epsilon)

    def configure_assignment(self, terrain, command):
        epsilon, labels = balanced_assignment(terrain, command, self.epsilon_seed, self.epsilon_bins)
        self.epsilon.copy_(epsilon.to(self.device))
        self.epsilon_labels = labels.to(self.device)
        self.pid = QuietPawPID(self.epsilon, **self.pid_cfg)
        self.assignment = dict(seed=self.epsilon_seed, bins=self.epsilon_bins,
                               terrain=terrain.cpu().tolist(), command=command.cpu().tolist(),
                               epsilon=epsilon.flatten().tolist(), labels=labels.tolist())

    def act(self, obs, critic_obs):
        ac, tr = self.actor_critic, self.transition
        tr.actions = ac.act(obs, self.epsilon).detach()
        tr.values = ac.evaluate(critic_obs, self.epsilon).detach()
        tr.cost_value = ac.evaluate_cost(critic_obs, self.epsilon).detach()
        tr.actions_log_prob = ac.get_actions_log_prob(tr.actions).detach()
        tr.action_mean, tr.action_sigma = ac.action_mean.detach(), ac.action_std.detach()
        # Snapshot observations and conditions before simulator reset/mutation.
        tr.observations, tr.critic_observations = obs.clone(), critic_obs.clone()
        tr.epsilon = self.epsilon.clone()
        return tr.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        tr = self.transition
        cost = infos["quietpaw"]["noise_cost"].to(self.device).reshape(-1, 1)
        if cost.shape != self.epsilon.shape or not torch.isfinite(cost).all() or ((cost < 0) | (cost > 1)).any():
            raise ValueError("noise_cost must be finite, per environment, and within [0, 1]")
        done = dones.to(self.device).bool().reshape(-1, 1)
        timeout = infos.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1, 1)
        if (timeout & ~done).any():
            raise ValueError("A timeout must also terminate its rollout segment")
        failure = infos["quietpaw"].get("failure", torch.zeros_like(dones)).to(self.device).bool().reshape(-1, 1)
        timeout = timeout & ~failure  # physical failure wins if both flags occur
        # next_critic_obs has the PRE-RESET terminal observation restored by the
        # inherited HIM runner. A reset observation must never bootstrap cost.
        with torch.no_grad():
            terminal_cost_value = self.actor_critic.evaluate_cost(next_critic_obs, tr.epsilon)
            terminal_reward_value = self.actor_critic.evaluate(next_critic_obs, tr.epsilon)
        tr.noise_cost = cost.clone()
        tr.cost_reward = (1 - self.gamma) * cost + self.gamma * timeout * terminal_cost_value
        self.episode_cost += self.episode_discount * (1 - self.gamma) * cost
        self.episode_discount *= self.gamma
        completed = self.episode_cost + self.episode_discount * timeout * terminal_cost_value
        self.completed_cost += torch.where(done, completed, torch.zeros_like(completed))
        self.completed_count += done
        self.episode_cost[done] = 0.
        self.episode_discount[done] = 1.
        # Keep old tr.values = V(s_t) for PPO clipping/GAE. Both timeout targets
        # use the same terminal observation and the transition's epsilon.
        tr.next_critic_observations = next_critic_obs.clone()
        tr.rewards = rewards.clone() + self.gamma * (timeout * terminal_reward_value).squeeze(-1)
        tr.dones = dones
        self.storage.add_transitions(tr)
        tr.clear()
        self.actor_critic.reset(dones)
        metrics = {"reward": rewards, **infos["quietpaw"]}
        for name, value in metrics.items():
            value = value.to(self.device).reshape(-1, 1).float()
            if name not in self.metric_sums:
                self.metric_sums[name] = torch.zeros_like(self.epsilon)
            self.metric_sums[name] += value
        self.metric_steps += 1
        self.success_count += timeout

    def compute_returns(self, last_critic_obs):
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_critic_obs, self.epsilon)
            last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs, self.epsilon)
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        self.lambda_used = self.pid.multiplier.clone()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam, self.lambda_used)

    def value_loss(self, prediction, old_value, target):
        error = (prediction - target).square()
        if self.use_clipped_value_loss:
            clipped = old_value + (prediction - old_value).clamp(-self.clip_param, self.clip_param)
            error = torch.maximum(error, (clipped - target).square())
        return error.mean()

    @staticmethod
    def policy_loss(log_prob, old_log_prob, advantage, clip_param):
        ratio = torch.exp(log_prob - old_log_prob)
        return torch.maximum(-advantage * ratio,
                             -advantage * ratio.clamp(1 - clip_param, 1 + clip_param)).mean()

    def update(self):
        # This mirrors HIM's PPO update order. The only learning changes are
        # epsilon inputs, combined advantages and the additional cost critic.
        totals = torch.zeros(5, device=self.device)
        updates = 0
        for batch in self.storage.cmdp_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            ac = self.actor_critic
            ac.act(batch["observations"], batch["epsilon"])
            log_prob = ac.get_actions_log_prob(batch["actions"])
            value = ac.evaluate(batch["critic"], batch["epsilon"])
            cost_value = ac.evaluate_cost(batch["critic"], batch["epsilon"])
            mu, sigma, entropy = ac.action_mean, ac.action_std, ac.entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = torch.sum(torch.log(sigma / batch["sigma"] + 1.e-5) +
                                   (batch["sigma"].square() + (batch["mu"] - mu).square()) /
                                   (2 * sigma.square()) - 0.5, dim=-1).mean()
                    if kl > self.desired_kl * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            # EXACT HIM call: one auxiliary update per PPO minibatch, unmodified
            # 270-D history / 238-D next critic observation, same adaptive LR.
            estimation_loss, swap_loss = ac.estimator.update(
                batch["observations"], batch["next_critic"], lr=self.learning_rate)

            advantage = batch["advantages"].squeeze(-1)
            surrogate_loss = self.policy_loss(log_prob, batch["log_prob"].squeeze(-1), advantage, self.clip_param)
            reward_value_loss = self.value_loss(value, batch["values"], batch["returns"])
            cost_value_loss = self.value_loss(cost_value, batch["cost_value"], batch["cost_returns"])
            loss = (surrogate_loss + self.value_loss_coef * reward_value_loss +
                    self.cost_value_loss_coef * cost_value_loss - self.entropy_coef * entropy.mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite constrained PPO loss")
            # Clears estimator gradients left by its own optimizer, just as HIM
            # does. Detached encoder outputs prevent PPO from updating HIM.
            self.optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None for p in ac.estimator.parameters()):
                raise RuntimeError("PPO gradients entered HIMEstimator")
            nn.utils.clip_grad_norm_(ac.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals += torch.tensor([reward_value_loss.item(), surrogate_loss.item(),
                                    estimation_loss, swap_loss, cost_value_loss.item()], device=self.device)
            updates += 1

        valid = self.completed_count > 0
        measured_cost = self.completed_cost / self.completed_count.clamp_min(1)
        # Algorithm 1: advantages use the old lambda; one PID update per
        # collection/update cycle supplies the NEXT cycle's lambda. No minibatch
        # updates and no reuse of stale episode statistics.
        self.pid.update(measured_cost, valid)
        self.diagnostics = {
            "cost_value_loss": (totals[4] / updates).item(),
            "noise_cost_mean": self.storage.noise_cost.mean().item(),
            "lambda_used_mean": self.lambda_used.mean().item(),
            "lambda_next_mean": self.pid.multiplier.mean().item(),
            "lambda_next_max": self.pid.multiplier.max().item(),
            "pid_envs_updated": valid.sum().item(),
            "completed_episodes": self.completed_count.sum().item(),
            "budget_violation_mean": ((measured_cost - self.epsilon)[valid].clamp_min(0).mean().item()
                                      if valid.any() else 0.),
            "him_updates": updates,
        }
        for label in self.epsilon_labels.unique().tolist():
            mask = self.epsilon_labels == label
            prefix = "epsilon_bin_{}/".format(label)
            self.diagnostics[prefix + "epsilon_min"] = self.epsilon[mask].min().item()
            self.diagnostics[prefix + "epsilon_max"] = self.epsilon[mask].max().item()
            for name, total in self.metric_sums.items():
                self.diagnostics[prefix + name + "_mean"] = (total[mask].mean() / self.metric_steps).item()
            episodes = self.completed_count[mask].sum()
            self.diagnostics[prefix + "completed_episodes"] = episodes.item()
            self.diagnostics[prefix + "lambda_mean"] = self.pid.multiplier[mask].mean().item()
            if episodes > 0:
                self.diagnostics[prefix + "timeout_success_rate"] = (self.success_count[mask].sum() / episodes).item()
            observed = valid.squeeze(-1) & mask
            if observed.any():
                self.diagnostics[prefix + "budget_violation"] = (measured_cost - self.epsilon)[observed].clamp_min(0).mean().item()
        self.metric_sums.clear()
        self.metric_steps = 0
        self.success_count.zero_()
        self.completed_cost.zero_()
        self.completed_count.zero_()
        self.iterations += 1
        self.storage.clear()
        # Preserve inherited HIM runner's logging interface.
        return tuple((totals[:4] / updates).tolist())
