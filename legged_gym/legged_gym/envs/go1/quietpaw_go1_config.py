"""Go1 Stage 0 entry point; all environment and locomotion settings inherit."""

from .go1_config import Go1RoughCfg, Go1RoughCfgPPO


class QuietPawGo1Cfg(Go1RoughCfg):
    """The environment remains the original LeggedRobot with Go1 settings."""


class QuietPawGo1CfgPPO(Go1RoughCfgPPO):
    runner_class_name = "QuietPawRunner"

    class quietpaw:
        stage = 0
        epsilon = None
        noise_cost = None

    class runner(Go1RoughCfgPPO.runner):
        policy_class_name = "QuietPawHIMActorCritic"
        algorithm_class_name = "QuietPawHIMPPO"
        experiment_name = "quietpaw_go1_stage0"
