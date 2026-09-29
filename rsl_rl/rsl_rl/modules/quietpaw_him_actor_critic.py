"""Stage 0 extension point: exactly the original HIM networks and inputs."""

from .him_actor_critic import HIMActorCritic


class QuietPawHIMActorCritic(HIMActorCritic):
    """No conditioning or cost critic yet: actor 64, critic 238, action 12.

    Stage 1 will introduce epsilon as an explicit side input. In particular,
    neither the history input nor HIMEstimator is extended in Stage 0.
    """
