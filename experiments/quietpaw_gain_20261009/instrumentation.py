"""Read-only diagnostics around the existing constrained update."""
import json
from pathlib import Path
import torch
from torch.distributions import Normal


def gradient_blocks(ac, storage, multipliers, clip):
    reward = storage.reward_advantages / (1 + multipliers)
    cost = storage.cost_advantages * multipliers / (1 + multipliers)
    scale = storage.combined_advantages.std() + 1e-8
    reward = (reward-reward.mean()) / scale
    cost = (cost-cost.mean()) / scale
    params = tuple(ac.actor.parameters()) + (ac.std,)
    blocks = sorted(set([0, max(0,storage.num_transitions_per_env//2-1),
                         max(0,storage.num_transitions_per_env-2)]))
    records = []
    for start in blocks:
        take = slice(start,start+2)
        obs = storage.observations[take].flatten(0,1)
        eps = storage.epsilon[take].flatten(0,1)
        actions = storage.actions[take].flatten(0,1)
        old = storage.actions_log_prob[take].flatten().detach()
        r,c = reward[take].flatten().detach(),cost[take].flatten().detach()
        with torch.enable_grad():
            mu=ac.act_inference(obs,eps)
            ratio=(Normal(mu,mu*0+ac.std).log_prob(actions).sum(-1)-old).exp()
            joint=r-c
            gate=~(((joint>=0)&(ratio>1+clip))|((joint<0)&(ratio<1-clip)))
            row=dict(start=start,samples=len(obs),clipped_fraction=(~gate).float().mean().item())
            for label,mask in [('before_clip',torch.ones_like(ratio)),('after_clip',gate.float().detach())]:
                gr=torch.autograd.grad((-ratio*r*mask).mean(),params,retain_graph=True)
                gc=torch.autograd.grad((ratio*c*mask).mean(),params,retain_graph=True)
                for group,subset in [('actor',slice(None,-1)),('std',slice(-1,None))]:
                    a,b=gr[subset],gc[subset]
                    nr=sum(x.square().sum() for x in a).sqrt()
                    nc=sum(x.square().sum() for x in b).sqrt()
                    dot=sum((x*y).sum() for x,y in zip(a,b))
                    row[label+'_'+group]=dict(reward_norm=nr.item(),cost_norm=nc.item(),
                        ratio=(nc/nr.clamp_min(1e-12)).item(),
                        cosine=(dot/(nr*nc).clamp_min(1e-12)).item(),
                        cosine_defined=bool(nr>1e-12 and nc>1e-12))
            records.append(row)
    return records


def install(alg, output, endpoint=300):
    """Return captured PID input list; inherited update still does all learning."""
    path=Path(output); path.mkdir(parents=True,exist_ok=True)
    ac=alg.actor_critic
    original_update=alg.update
    pid_inputs=[]

    def instrumented_update():
        st=alg.storage; iteration=alg.iterations
        selected=iteration%50==0 or iteration==endpoint-1
        record=dict(iteration=iteration,minibatches=[])
        pid_inputs.append(dict(cost=(alg.completed_cost/alg.completed_count.clamp_min(1)).detach().cpu().clone(),
                               valid=(alg.completed_count>0).detach().cpu().clone()))
        if selected:
            record['gradients_before']=gradient_blocks(ac,st,alg.lambda_used,alg.clip_param)
        ids=torch.linspace(0,st.observations.flatten(0,1).shape[0]-1,256,device=alg.device).long()
        obs,eps=st.observations.flatten(0,1)[ids],st.epsilon.flatten(0,1)[ids]
        old_mu,old_sigma=st.mu.flatten(0,1)[ids],st.sigma.flatten(0,1)[ids]

        @torch.no_grad()
        def policy():
            return ac.act_inference(obs,eps).clone(),ac.std.expand_as(old_mu).clone()

        @torch.no_grad()
        def kl(p,q):
            mu,s=p; m,t=q
            return (torch.log(t/s)+(s.square()+(mu-m).square())/(2*t.square())-.5).sum(-1).mean().item()

        original_him=ac.estimator.update
        original_step=alg.optimizer.step
        original_clip=torch.nn.utils.clip_grad_norm_
        def him_update(*args,**kwargs):
            before=policy()
            result=original_him(*args,**kwargs)
            after=policy()
            record['minibatches'].append(dict(lr=alg.learning_rate,him_local_kl=kl(before,after),
                                              after_him_kl=kl((old_mu,old_sigma),after)))
            return result

        def clip(parameters,*args,**kwargs):
            parameters=list(parameters)
            if any(p is ac.std for p in parameters):
                groups=dict(actor=ac.actor.parameters(),std=[ac.std],reward_critic=ac.critic.parameters(),
                            cost_critic=ac.cost_critic.parameters())
                norms={k:sum(p.grad.square().sum().item() for p in ps if p.grad is not None)**.5
                       for k,ps in groups.items()}
                record['minibatches'][-1]['pre_global_clip_norms']=norms
            return original_clip(parameters,*args,**kwargs)

        def step(*args,**kwargs):
            before=policy()
            result=original_step(*args,**kwargs)
            after=policy()
            record['minibatches'][-1].update(ppo_local_kl=kl(before,after),
                                            after_ppo_kl=kl((old_mu,old_sigma),after))
            return result

        ac.estimator.update=him_update
        alg.optimizer.step=step
        torch.nn.utils.clip_grad_norm_=clip
        try:
            result=original_update()
        finally:
            ac.estimator.update=original_him
            alg.optimizer.step=original_step
            torch.nn.utils.clip_grad_norm_=original_clip
        if selected:
            record['gradients_after']=gradient_blocks(ac,st,alg.lambda_used,alg.clip_param)
        record['lambda_mean']=alg.pid.multiplier.mean().item()
        with (path/'mechanism.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        if iteration==endpoint-1:
            torch.save(dict(epsilon=alg.epsilon.cpu(),inputs=pid_inputs),path/'pid_inputs.pt')
        return result

    alg.update=instrumented_update
    return pid_inputs
