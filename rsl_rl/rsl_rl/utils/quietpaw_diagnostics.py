"""Read-only probes of collected training data; never sample or populate .grad."""

import torch
from torch.distributions import Normal


@torch.no_grad()
def advantage_diagnostics(storage, multipliers, labels):
    reward = storage.reward_advantages
    cost = storage.cost_advantages
    reward_term = reward / (1 + multipliers)
    cost_term = multipliers * cost / (1 + multipliers)
    result = {}
    for label in labels.unique().tolist():
        mask = labels == label
        prefix = "epsilon_bin_{}/".format(label)
        rstd = reward_term[:, mask].std(unbiased=False)
        cstd = cost_term[:, mask].std(unbiased=False)
        result[prefix + "reward_advantage_std"] = reward[:, mask].std(unbiased=False).item()
        result[prefix + "cost_advantage_std"] = cost[:, mask].std(unbiased=False).item()
        result[prefix + "weighted_reward_advantage_std"] = rstd.item()
        result[prefix + "weighted_cost_advantage_std"] = cstd.item()
        result[prefix + "weighted_cost_reward_advantage_ratio"] = (cstd / rstd.clamp_min(1e-12)).item()
    return result


def actor_gradient_diagnostics(model, storage, multipliers, labels, probe_steps=8):
    """Pre-clip score-function loss components on fixed samples at theta_old.

    Components include 1/(1+lambda); no entropy or value losses are included.
    These are per-bin finite-sample gradients, not post-clipping update norms.
    """
    steps = min(probe_steps, storage.num_transitions_per_env)
    history = storage.observations[:steps].flatten(0, 1)
    epsilon = storage.epsilon[:steps].flatten(0, 1)
    actions = storage.actions[:steps].flatten(0, 1)
    reward = (storage.reward_advantages[:steps] / (1 + multipliers)).flatten().detach()
    cost = (multipliers * storage.cost_advantages[:steps] / (1 + multipliers)).flatten().detach()
    condition = labels.repeat(steps)
    parameters = tuple(model.actor.parameters()) + (model.std,)
    result = {}
    with torch.enable_grad():
        mean = model.act_inference(history, epsilon)
        distribution = Normal(mean, mean * 0. + model.std)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        bins = labels.unique().tolist()
        for index, label in enumerate(bins):
            mask = condition == label
            gr = torch.autograd.grad((-log_prob[mask] * reward[mask]).mean(),
                                     parameters, retain_graph=True)
            gc = torch.autograd.grad((log_prob[mask] * cost[mask]).mean(),
                                     parameters, retain_graph=index < len(bins) - 1)
            rnorm = torch.stack([g.square().sum() for g in gr]).sum().sqrt()
            cnorm = torch.stack([g.square().sum() for g in gc]).sum().sqrt()
            dot = torch.stack([(r * c).sum() for r, c in zip(gr, gc)]).sum()
            prefix = "epsilon_bin_{}/".format(label)
            result[prefix + "actor_reward_gradient_norm"] = rnorm.item()
            result[prefix + "actor_cost_gradient_norm"] = cnorm.item()
            result[prefix + "actor_cost_reward_gradient_ratio"] = (cnorm / rnorm.clamp_min(1e-12)).item()
            result[prefix + "actor_reward_cost_loss_gradient_cosine"] = (
                dot / (rnorm * cnorm).clamp_min(1e-12)).item()
    result["actor_gradient_probe_steps"] = steps
    return result
