"""Stage 0 runner reusing the complete HIM collection and learning loop."""

from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.modules.quietpaw_him_actor_critic import QuietPawHIMActorCritic
from rsl_rl.algorithms.quietpaw_him_ppo import QuietPawHIMPPO


class QuietPawRunner(HIMOnPolicyRunner):
    def __init__(self, env, train_cfg, *args, **kwargs):
        scaffold = train_cfg.get("quietpaw", {})
        if scaffold.get("stage", 0) != 0:
            raise NotImplementedError("Only QuietPaw Stage 0 is implemented.")
        for name in ("epsilon", "noise_cost"):
            if scaffold.get(name) is not None:
                raise NotImplementedError(
                    "{} is an unset Stage 0 placeholder, not an active "
                    "training parameter.".format(name)
                )
        self.epsilon = None
        self.noise_cost = None
        self.cost_value = None
        super().__init__(env, train_cfg, *args, **kwargs)
        print("QuietPaw Stage 0 scaffold: original HIM PPO; actor input 64; "
              "epsilon/noise_cost/cost_value unavailable; no cost optimization.")

    def _resolve_policy_class(self, name):
        if name == "QuietPawHIMActorCritic":
            return QuietPawHIMActorCritic
        return super()._resolve_policy_class(name)

    def _resolve_algorithm_class(self, name):
        if name == "QuietPawHIMPPO":
            return QuietPawHIMPPO
        return super()._resolve_algorithm_class(name)
