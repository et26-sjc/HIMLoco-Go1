"""Stage 1 reuses HIM collection and exposes explicit conditional inference."""

import torch
import wandb
import json
from pathlib import Path
from .quietpaw_runner import QuietPawRunner
from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic
from rsl_rl.algorithms.quietpaw_constrained_ppo import QuietPawConstrainedPPO
from rsl_rl.utils.quietpaw_him_warmstart import initialize_from_him_checkpoint


class QuietPawStage1Runner(QuietPawRunner):
    def __init__(self, env, train_cfg, *args, **kwargs):
        if train_cfg.get("quietpaw", {}).get("stage") != 1:
            raise ValueError("QuietPawStage1Runner requires stage=1")
        self.contract = {
            "version": 3,
            "reward_bootstrap": "terminal_critic_observation",
            "epsilon_assignment": "joint_cohort_stratified_full_grid_v1",
            "noise_definition": "normalized_complement_v2",
            "budget": "normalized_discounted_episode_return",
            "epsilon_semantics": "absolute_cost_budget",
            "epsilon_range": {
                "minimum": train_cfg["algorithm"]["epsilon_min"],
                "maximum": train_cfg["algorithm"]["epsilon_max"],
            },
            "sampling": "policy_step_pre_reset",
            "algorithm": train_cfg["algorithm"],
        }
        fixed = train_cfg["algorithm"].get("epsilon_fixed")
        if fixed is not None:
            self.contract["epsilon_assignment"] = "fixed_budget_v1"
            self.contract["epsilon_range"] = {"minimum": fixed, "maximum": fixed}
        noise_cfg = getattr(env.cfg, "noise_cost", None)
        if noise_cfg is None or noise_cfg.definition != self.contract["noise_definition"]:
            raise ValueError("Stage 1 requires the QuietPaw noise measurement environment")
        self.contract["noise_parameters"] = {
            name: getattr(noise_cfg, name) for name in
            ("sigma_force", "sigma_velocity", "contact_threshold", "force_weight", "impact_weight")}
        # Bypass the Stage 0 guard, while retaining its class-selection extension.
        HIMOnPolicyRunner.__init__(self, env, train_cfg, *args, **kwargs)
        terrain = getattr(env, "terrain_types", torch.zeros(env.num_envs, dtype=torch.long))
        command = (torch.arange(env.num_envs) < env.num_envs * 0.2).long()
        self.alg.configure_assignment(terrain, command)
        self.initialization = None
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.log_dir) / "epsilon_assignment.json").write_text(
                json.dumps(self.alg.assignment, indent=2))
        print("QuietPaw Stage 1 uses documented reproduction assumptions: "
              "normalized_complement_v2, velocity-weighted cost, direct epsilon budget, per-env PID. "
              "No value decomposition; see docs/quietpaw_budget_velocity.md.")

    def _resolve_policy_class(self, name):
        if name == "QuietPawConditionalActorCritic":
            return QuietPawConditionalActorCritic
        return super()._resolve_policy_class(name)

    def _resolve_algorithm_class(self, name):
        if name == "QuietPawConstrainedPPO":
            return QuietPawConstrainedPPO
        return super()._resolve_algorithm_class(name)

    def _evaluate_logging_values(self, critic_obs):
        return self.alg.actor_critic.evaluate(critic_obs, self.alg.epsilon)

    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        for name, value in self.alg.diagnostics.items():
            self.writer.add_scalar("QuietPaw/" + name, value, locs["it"])
        if self.use_wandb:
            wandb.log({"trainer/iteration": locs["it"],
                       **{"quietpaw/" + k: v for k, v in self.alg.diagnostics.items()}})
        print("QuietPaw:", self.alg.diagnostics)

    def _checkpoint_state(self, infos=None):
        state = super()._checkpoint_state(infos)
        state["iter"] = self.alg.iterations
        state["quietpaw"] = {
            "contract": self.contract,
            "pid": self.alg.pid.state_dict(),
            "learning_rate": self.alg.learning_rate,
            "assignment": self.alg.assignment,
            "initialization": self.initialization,
        }
        # PhysX state is not saved by HIM. Incomplete episode accumulators are
        # intentionally not resumed into fresh environments.
        return state

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device)
        if "quietpaw" not in state:
            raise ValueError("Stage 0/HIM checkpoint cannot resume a conditioned model. "
                             "An explicit weight migration is required.")
        saved = state["quietpaw"]
        if saved["contract"] != self.contract:
            raise ValueError("QuietPaw checkpoint contract mismatch (cost/budget/algorithm)")
        if load_optimizer and not torch.equal(saved["pid"]["epsilon"].to(self.alg.epsilon), self.alg.epsilon):
            raise ValueError("Training resume requires the same environment count and epsilon grid")
        if load_optimizer and saved["assignment"] != self.alg.assignment:
            raise ValueError("Training resume requires identical terrain/command/epsilon mapping")
        self.alg.actor_critic.load_state_dict(state["model_state_dict"])
        self.initialization = saved.get("initialization")
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(state["estimator_optimizer_state_dict"])
            self.alg.pid.load_state_dict(saved["pid"])
            self.alg.learning_rate = saved["learning_rate"]
            self.alg.iterations = state["iter"]
            self.current_learning_iteration = state["iter"]
            self.alg.episode_cost.zero_()
            self.alg.episode_discount.fill_(1.)
            self.alg.completed_cost.zero_()
            self.alg.completed_count.zero_()
        return state["infos"]

    def initialize_from_him(self, checkpoint_path):
        if self.alg.iterations != 0 or self.current_learning_iteration != 0:
            raise ValueError("HIM warm-start is only valid before Stage 1 training")
        self.initialization = initialize_from_him_checkpoint(
            self.alg.actor_critic, checkpoint_path)
        if self.log_dir is not None:
            (Path(self.log_dir) / "him_warmstart.json").write_text(
                json.dumps(self.initialization, indent=2))
        return self.initialization

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        # Call policy(history_270, epsilon_Nx1). No privileged inputs required.
        return self.alg.actor_critic.act_inference
