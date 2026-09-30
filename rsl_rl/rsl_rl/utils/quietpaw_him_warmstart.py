"""Explicit weight-only transfer from an original HIM checkpoint to Stage 1.

This is an experimental initialization, not the curriculum reported by QuietPaw.
The new epsilon columns start at zero and the cost critic starts with a zero
output. PPO, HIM and PID optimizer states are intentionally fresh.
"""

import hashlib
from pathlib import Path

import torch

from rsl_rl.modules.quietpaw_conditional_actor_critic import QuietPawConditionalActorCritic


def initialize_from_him_checkpoint(model, checkpoint_path):
    if not isinstance(model, QuietPawConditionalActorCritic):
        raise TypeError("HIM warm-start requires the Stage 1 conditional actor-critic")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(str(path), map_location="cpu")
    if "quietpaw" in checkpoint:
        raise ValueError("Warm-start source must be an original HIM checkpoint")
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    expected = {key for key in target if not key.startswith("cost_critic.")}
    if set(source) != expected:
        raise ValueError("HIM checkpoint and Stage 1 model have different parameter names")

    expanded = {"actor.0.weight", "critic.0.weight"}
    for key in expected:
        old, new = source[key], target[key]
        if key in expanded:
            if old.ndim != 2 or new.shape != (old.shape[0], old.shape[1] + 1):
                raise ValueError("Unexpected epsilon expansion shape: " + key)
            new[:, :-1].copy_(old)
            new[:, -1].zero_()
        elif old.shape == new.shape:
            new.copy_(old)
        else:
            raise ValueError("Unexpected HIM parameter shape: " + key)
        if not torch.isfinite(new).all():
            raise ValueError("Non-finite HIM parameter: " + key)

    for key in target:
        if not key.startswith("cost_critic."):
            continue
        reward_key = "critic." + key[len("cost_critic."):]
        if key.startswith("cost_critic.6."):
            target[key].zero_()
        else:
            target[key].copy_(target[reward_key])
    model.load_state_dict(target, strict=True)
    return {
        "method": "him_weights_to_conditional_stage1_v1",
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_filename": path.name,
        "source_checkpoint_iter_field": checkpoint.get("iter"),
        "optimizer_state": "fresh_PPO_HIM_PID",
        "epsilon_columns": "zero",
        "cost_critic_output": "zero",
    }
