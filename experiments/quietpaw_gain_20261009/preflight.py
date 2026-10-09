"""Check PID scaling on stored episode data against a separate recursion."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    import isaacgym
    import torch
    from rsl_rl.algorithms.quietpaw_pid import QuietPawPID
    source=ROOT/'legged_gym/logs/quietpaw_followup_20261009'
    values=[]
    for seed in (111,112):
        data=json.loads((source/f'seed4_low_eval{seed}/result.json').read_text())['episodes']
        values.append((torch.tensor(data['discounted_cost'])+torch.tensor(data['timeout_tail'])).reshape(-1,1))
    eps=torch.full_like(values[0],.037646)
    outputs={}
    for factor in (0,1,100,1000):
        pid=QuietPawPID(eps,kp=factor,ki=.1*factor,kd=.1*factor)
        integral=torch.zeros_like(eps); p=torch.zeros_like(eps); ema=eps.clone()
        trace=[]
        for cost in values:
            valid=torch.ones_like(eps,dtype=torch.bool)
            error=cost-eps
            integral=torch.maximum(torch.zeros_like(eps),integral+.1*factor*error)
            p=.95*p+.05*error
            new=.95*ema+.05*cost
            expected=(integral+factor*p+.1*factor*(new-ema).clamp_min(0)).clamp_min(0)
            pid.update(cost,valid)
            torch.testing.assert_close(pid.multiplier,expected)
            trace.append(pid.multiplier.clone());ema=new
        outputs[factor]=torch.stack(trace)
    for factor in (0,100,1000):
        torch.testing.assert_close(outputs[factor],outputs[1]*factor)
    # The integral must decrease for a sufficiently small but negative error;
    # invalid environments must keep every state field unchanged.
    pid=QuietPawPID(eps); valid=torch.ones_like(eps,dtype=torch.bool)
    pid.update(eps+.01,valid); previous=pid.integral.clone()
    pid.update(eps-.001,valid)
    assert (pid.integral<previous).all()
    before=pid.state_dict();pid.update(eps+.1,~valid)
    assert all(torch.equal(v,pid.state_dict()[k]) for k,v in before.items())
    out=ROOT/'legged_gym/logs/quietpaw_gain_20261009'
    out.mkdir(parents=True,exist_ok=True)
    report=dict(description='Arithmetic replay of two recorded evaluation vectors, not a changed-policy prediction',
                multiplier_means={str(k):v.mean(dim=(1,2)).tolist() for k,v in outputs.items()},
                integral_decreases=True,invalid_event_mask_preserved=True)
    (out/'preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
