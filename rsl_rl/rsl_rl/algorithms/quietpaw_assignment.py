"""Seeded stratification of the complete epsilon grid over joint cohorts."""

import math

import torch


def validate_epsilon_range(minimum, maximum):
    if not (math.isfinite(minimum) and math.isfinite(maximum)
            and 0 <= minimum < maximum <= 1):
        raise ValueError("epsilon range must satisfy 0 <= minimum < maximum <= 1")


def balanced_assignment(terrain, command, seed, bins=8, minimum=0.0, maximum=1.0):
    validate_epsilon_range(minimum, maximum)
    terrain, command = terrain.cpu().long(), command.cpu().long()
    count = len(terrain)
    if bins < 2 or count < bins or command.shape != terrain.shape:
        raise ValueError("Invalid epsilon assignment dimensions/bins")
    generator = torch.Generator().manual_seed(seed)
    cohorts = torch.stack((terrain, command), dim=1)
    groups = torch.unique(cohorts, dim=0)
    labels = torch.empty(count, dtype=torch.long)
    offset = 0
    for group in groups:
        ids = (cohorts == group).all(dim=1).nonzero().flatten()
        if len(ids) < bins:
            raise ValueError("Each terrain/command cohort needs >= {} environments; cohort {} has {}. "
                             "Increase num_envs (Go1: 160 or more), not change terrain.".format(bins, group.tolist(), len(ids)))
        ids = ids[torch.randperm(len(ids), generator=generator)]
        labels[ids] = torch.arange(offset, offset + len(ids)) % bins
        offset += len(ids)
    epsilon = torch.empty(count)
    grid = torch.linspace(minimum, maximum, count)
    # Cyclic labels and tensor_split have identical global bin counts.
    for label, values in enumerate(torch.tensor_split(grid, bins)):
        ids = (labels == label).nonzero().flatten()
        epsilon[ids] = values[torch.randperm(len(values), generator=generator)]
    return epsilon.unsqueeze(1), labels
