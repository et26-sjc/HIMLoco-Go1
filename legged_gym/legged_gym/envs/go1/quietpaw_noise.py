"""Explicit monotone interpretation of QuietPaw's ambiguous Eq. (1).

The v2 experiment gives touchdown velocity more weight than support force.
This is NOT verified author code. Inputs are world-frame force (N) and
pre-step velocity (m/s); timing and masks are unchanged.
"""

import math

import torch


def noise_cost_components(force, previous_velocity, previous_contact, valid,
                          sigma_force=50., sigma_velocity=0.5, contact_threshold=1.,
                          force_weight=0.2, impact_weight=0.8):
    parameters = (sigma_force, sigma_velocity, contact_threshold, force_weight, impact_weight)
    if (not all(math.isfinite(value) for value in parameters)
            or sigma_force <= 0 or sigma_velocity <= 0 or contact_threshold < 0
            or min(force_weight, impact_weight) < 0
            or not math.isclose(force_weight + impact_weight, 1., abs_tol=1e-8)):
        raise ValueError("Noise scales must be positive and nonnegative weights must sum to one")
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    contact = magnitude > contact_threshold
    touchdown = contact & ~previous_contact & valid.unsqueeze(-1)
    downward_speed = (-previous_velocity[..., 2]).clamp_min(0)
    force_component = -torch.expm1(-magnitude / sigma_force) * contact
    impact_component = -torch.expm1(-downward_speed.square() / sigma_velocity) * touchdown
    # Four-foot average has a fixed denominator: changing contact count does
    # not renormalize away the number of impacts. No dt factor in raw cost.
    force_cost = force_weight * force_component.mean(dim=-1)
    impact_cost = impact_weight * impact_component.mean(dim=-1)
    return force_cost, impact_cost, contact, touchdown


def noise_cost(force, previous_velocity, previous_contact, valid,
               sigma_force=50., sigma_velocity=0.5, contact_threshold=1.,
               force_weight=0.2, impact_weight=0.8):
    force_cost, impact_cost, contact, touchdown = noise_cost_components(
        force, previous_velocity, previous_contact, valid,
        sigma_force, sigma_velocity, contact_threshold, force_weight, impact_weight)
    return force_cost + impact_cost, contact, touchdown
