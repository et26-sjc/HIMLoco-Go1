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


class QuietPawStage1Go1Cfg(QuietPawGo1Cfg):
    class noise_cost:
        # Explicit reproduction assumptions; not verified author code.
        definition = "normalized_complement_v1"
        sigma_force = 50.0
        sigma_velocity = 0.5
        contact_threshold = 1.0


class QuietPawStage1Go1CfgPPO(QuietPawGo1CfgPPO):
    runner_class_name = "QuietPawStage1Runner"

    class quietpaw:
        stage = 1

    class algorithm(Go1RoughCfgPPO.algorithm):
        cost_value_loss_coef = 1.0
        epsilon_seed = 1
        epsilon_bins = 8

        class pid:
            # QuietPaw specifies PID but does not report its gains.
            kp = 1.0
            ki = 0.1
            kd = 0.1
            ema = 0.95
            initial = 0.0

    class runner(QuietPawGo1CfgPPO.runner):
        policy_class_name = "QuietPawConditionalActorCritic"
        algorithm_class_name = "QuietPawConstrainedPPO"
        experiment_name = "quietpaw_go1_stage1"
