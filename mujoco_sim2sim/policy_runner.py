from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PolicyRunner:
    policy_path: Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()

    def __call__(self, obs_history: np.ndarray) -> np.ndarray:
        if obs_history.ndim != 2:
            raise ValueError(f"obs_history must be 2D (B, obs_dim), got {obs_history.shape}")

        obs_tensor = torch.as_tensor(obs_history, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.policy(obs_tensor)
        out = actions.detach().cpu().numpy()
        return out[0] if out.shape[0] == 1 else out
