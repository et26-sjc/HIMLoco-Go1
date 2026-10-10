"""V1 quiet adaptation settings; the original go1 task is unchanged."""

from .go1_config import Go1RoughCfg, Go1RoughCfgPPO


class Go1QuietCfg(Go1RoughCfg):
    class env(Go1RoughCfg.env):
        num_envs = 256

    class terrain(Go1RoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        # Preserve the baseline's 238-D privileged critic input on flat ground.
        measure_heights = True

    class commands(Go1RoughCfg.commands):
        curriculum = False
        heading_command = False

    class domain_rand(Go1RoughCfg.domain_rand):
        push_robots = False
        disturbance = False

    class sim(Go1RoughCfg.sim):
        class physx(Go1RoughCfg.sim.physx):
            # The 4096-environment baseline reserves 2**23 contact pairs.
            # V1 uses 256 robots and a separate evaluation process. Reduce the
            # capacity reservation without changing contact solver settings.
            max_gpu_contact_pairs = 2**20

    class rewards(Go1RoughCfg.rewards):
        class scales(Go1RoughCfg.rewards.scales):
            # Actual weight is applied by _reward_quiet_touchdown. This fixed
            # nonzero scale registers the term even during the zero-weight ramp.
            quiet_touchdown = -1.0

    class quiet:
        training_speeds = [0.3, 0.6, 0.9]
        contact_enter_force = 2.0
        contact_exit_force = 0.5
        min_air_time = 0.01
        startup_time = 0.5
        impact_window = 0.02
        touchdown_reference = None
        target_weight = 0.0
        ramp_iterations = 50


class Go1QuietCfgPPO(Go1RoughCfgPPO):
    class algorithm(Go1RoughCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = "fixed"
        update_estimator = False

    class runner(Go1RoughCfgPPO.runner):
        policy_class_name = "QuietHIMActorCritic"
        experiment_name = "quiet_go1"
        max_iterations = 200
        save_interval = 50
