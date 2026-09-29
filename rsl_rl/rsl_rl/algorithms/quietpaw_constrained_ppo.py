"""Stage 1 PPO-Lagrangian with separate cost GAE and intact HIM updates."""

import torch
from torch import nn

from .quietpaw_him_ppo import QuietPawHIMPPO
from .quietpaw_pid import QuietPawPID
from rsl_rl.storage.quietpaw_cmdp_storage import QuietPawCMDPStorage


class QuietPawConstrainedPPO(QuietPawHIMPPO):
    def __init__(self, *args, cost_value_loss_coef=1.0, pid=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.gamma < 1:
            raise ValueError("Normalized discounted cost requires 0 < gamma < 1")
        self.cost_value_loss_coef = cost_value_loss_coef
        self.pid_cfg = pid or {}
        self.iterations = 0
        self.diagnostics = {}

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        if num_envs < 2:
            raise ValueError("Stage 1 training needs >=2 environments for the epsilon grid.")
        self.storage = QuietPawCMDPStorage(num_envs, num_transitions_per_env,
                                         actor_obs_shape, critic_obs_shape, action_shape, self.device)
        self.transition = QuietPawCMDPStorage.Transition()
        self.epsilon = torch.linspace(0., 1., num_envs, device=self.device).unsqueeze(1)
        self.pid = QuietPawPID(self.epsilon, **self.pid_cfg)
        self.episode_cost = torch.zeros_like(self.epsilon)
        self.episode_discount = torch.ones_like(self.epsilon)
        self.completed_cost = torch.zeros_like(self.epsilon)
        self.completed_count = torch.zeros_like(self.epsilon)

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
        # next_critic_obs has the PRE-RESET terminal observation restored by the
        # inherited HIM runner. A reset observation must never bootstrap cost.
        with torch.no_grad():
            terminal_cost_value = self.actor_critic.evaluate_cost(next_critic_obs, tr.epsilon)
        tr.noise_cost = cost.clone()
        tr.cost_reward = (1 - self.gamma) * cost + self.gamma * timeout * terminal_cost_value
        self.episode_cost += self.episode_discount * (1 - self.gamma) * cost
        self.episode_discount *= self.gamma
        completed = self.episode_cost + self.episode_discount * timeout * terminal_cost_value
        self.completed_cost += torch.where(done, completed, torch.zeros_like(completed))
        self.completed_count += done
        self.episode_cost[done] = 0.
        self.episode_discount[done] = 1.
        # Retains HIM's reward handling, timeout convention, next-state target
        # and reset path. Costs NEVER modify the locomotion reward.
        super().process_env_step(rewards, dones, infos, next_critic_obs)

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
        self.completed_cost.zero_()
        self.completed_count.zero_()
        self.iterations += 1
        self.storage.clear()
        # Preserve inherited HIM runner's logging interface.
        return tuple((totals[:4] / updates).tolist())
