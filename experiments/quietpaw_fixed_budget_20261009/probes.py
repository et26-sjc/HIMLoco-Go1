"""Exact joint-PPO clipping-gate gradient decomposition; no learning changes."""

import torch
from torch.distributions import Normal


def gradient_probe(model, storage, multipliers, clip=.2, steps=2):
    count = min(steps, storage.num_transitions_per_env)
    obs = storage.observations[:count].flatten(0, 1)
    eps = storage.epsilon[:count].flatten(0, 1)
    actions = storage.actions[:count].flatten(0, 1)
    old = storage.actions_log_prob[:count].flatten().detach()
    r = storage.reward_advantages / (1 + multipliers)
    c = multipliers * storage.cost_advantages / (1 + multipliers)
    scale = storage.combined_advantages.std() + 1e-8
    # Separate centering sums to the actual joint standardized advantage.
    ra = ((r - r.mean()) / scale)[:count].flatten().detach()
    ca = ((c - c.mean()) / scale)[:count].flatten().detach()
    unit = storage.cost_advantages[:count].flatten().detach() / scale
    params = tuple(model.actor.parameters()) + (model.std,)
    with torch.enable_grad():
        mean = model.act_inference(obs, eps)
        logp = Normal(mean, mean * 0 + model.std).log_prob(actions).sum(-1)
        ratio = (logp - old).exp()
        advantage = ra - ca
        gate = ~(((advantage >= 0) & (ratio > 1 + clip)) |
                 ((advantage < 0) & (ratio < 1 - clip)))
        result = {'samples': len(obs), 'clipping_active_fraction': (~gate).float().mean().item(),
                  'shared_advantage_scale': scale.item()}
        for name, mask in (('unclipped', torch.ones_like(ratio)), ('clipped', gate.float().detach())):
            gr = torch.autograd.grad((-ratio * ra * mask).mean(), params, retain_graph=True)
            gc = torch.autograd.grad((ratio * ca * mask).mean(), params, retain_graph=True)
            normr = sum(g.square().sum() for g in gr).sqrt()
            normc = sum(g.square().sum() for g in gc).sqrt()
            dot = sum((x * y).sum() for x, y in zip(gr, gc))
            cosine = dot / (normr * normc).clamp_min(1e-12)
            result.update({name + '_reward_norm': normr.item(), name + '_cost_norm': normc.item(),
                           name + '_cost_reward_ratio': (normc / normr.clamp_min(1e-12)).item(),
                           name + '_loss_gradient_cosine': cosine.item(),
                           name + '_cosine_defined': bool(normr > 1e-12 and normc > 1e-12)})
        gu = torch.autograd.grad((ratio * unit).mean(), params)
        result['unweighted_cost_norm'] = sum(g.square().sum() for g in gu).sqrt().item()
    return result
