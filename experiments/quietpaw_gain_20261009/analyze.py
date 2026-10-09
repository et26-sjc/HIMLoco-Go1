"""Audit every final endpoint and report all gain arms, including failures."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'legged_gym/logs/quietpaw_gain_20261009'


def main():
    import isaacgym
    import torch
    from rsl_rl.algorithms.quietpaw_pid import QuietPawPID
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['complete'] and all(v['complete'] for v in manifest['jobs'].values())
    for source,expected in manifest['sources'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
    records={}; checks={}; decisions={}
    metrics=['observed_discounted_cost','pid_bootstrapped_cost','timeout_tail_mean','survival',
             'tracking','cost_per_alive_second','alive_seconds','touchdown_rate','initial_cost_bias',
             'initial_cost_rmse','signed_violation']
    for seed in manifest['seeds']:
        initial=None
        for gain in manifest['gains']:
            name=f'seed{seed}_g{gain}';folder=OUT/name
            current=json.loads((folder/'initial.json').read_text())
            if initial is not None: assert initial==current,name
            initial=current
            checkpoint=torch.load(folder/'model_300.pt',map_location='cpu')
            assert checkpoint['iter']==300
            assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
            for k in ['optimizer_state_dict','estimator_optimizer_state_dict']:
                assert set(int(v['step']) for v in checkpoint[k]['state'].values())=={6000},(name,k)
            cfg=checkpoint['quietpaw']['contract']['algorithm']
            assert cfg['learning_rate']==1e-5 and cfg['schedule']=='adaptive'
            assert (cfg['pid']['kp'],cfg['pid']['ki'],cfg['pid']['kd'])==(gain,.1*gain,.1*gain)
            saved=checkpoint['quietpaw']['pid']
            inputs=torch.load(folder/'pid_inputs.pt',map_location='cpu')
            assert len(inputs['inputs'])==300
            pid=QuietPawPID(inputs['epsilon'],**cfg['pid'])
            for entry in inputs['inputs']:pid.update(entry['cost'],entry['valid'])
            for k,v in saved.items():torch.testing.assert_close(v,pid.state_dict()[k])
            checks[name]=dict(optimizer_steps=6000,him_steps=6000,pid_replay_matches=True,
                lambda_mean=saved['multiplier'].mean().item(),lambda_max=saved['multiplier'].max().item(),
                pid_updates_mean=saved['updates'].float().mean().item())
            ev=[json.loads((OUT/f'{name}_evaluate{s}/result.json').read_text()) for s in (111,112)]
            row={k:statistics.mean(e['summary'][k] for e in ev) for k in metrics}
            row['evaluation_seeds']={str(e['seed']):e['summary'] for e in ev}
            row['deployment']=json.loads((OUT/f'{name}_deploy11/result.json').read_text())['summary']
            assert 0 < row['deployment']['observer_pd_input_checks'] <= 4000
            mechanism=[json.loads(l) for l in (folder/'mechanism.jsonl').read_text().splitlines()]
            assert len(mechanism)==300 and all(len(m['minibatches'])==20 for m in mechanism)
            row['max_him_local_kl']=max(m['him_local_kl'] for r in mechanism for m in r['minibatches'])
            row['max_update_policy_kl']=max(m['after_ppo_kl'] for r in mechanism for m in r['minibatches'])
            row['max_lr']=max(m['lr'] for r in mechanism for m in r['minibatches'])
            row['final_gradient_blocks']=mechanism[-1]['gradients_before']
            records[name]=row
        baseline=records[f'seed{seed}_g0']
        for gain in manifest['gains'][1:]:
            row=records[f'seed{seed}_g{gain}']
            decision=dict(budget_met=row['observed_discounted_cost']<=manifest['epsilon'],
                alive_cost_lower=row['cost_per_alive_second']<baseline['cost_per_alive_second'],
                survival_preserved=row['survival']>=baseline['survival']-.01,
                tracking_preserved=row['tracking']<=1.1*baseline['tracking'])
            decision['passes']=all(decision.values())
            decisions[f'seed{seed}_g{gain}']=decision
    passing=[g for g in manifest['gains'][1:] if all(decisions[f'seed{s}_g{g}']['passes'] for s in manifest['seeds'])]
    report=dict(epsilon=manifest['epsilon'],checks=checks,records=records,decisions=decisions,
                gains_passing_both_seeds=passing,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# E3：低初始学习率下的 PID 增益对照', '',
           '初始 LR=1e-5，epsilon=0.037646；每组 300 轮。表中为两个独立随机评估 seed 的均值。', '',
           '| Seed | PID gain factor | 折扣 cost | 超预算量 | 存活率 | Tracking error | 每存活秒 cost | 平均 lambda |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in records.items():
        seed,gain=name[4:].split('_g')
        lines.append('| {} | {} | {:.6f} | {:+.6f} | {:.2%} | {:.5f} | {:.4f} | {:.4f} |'.format(
            seed,gain,r['observed_discounted_cost'],r['signed_violation'],r['survival'],r['tracking'],
            r['cost_per_alive_second'],checks[name]['lambda_mean']))
    lines+=['', '两个训练 seed 均通过预设判定的增益：'+str(passing)+'.', '',
            '判定同时要求预算满足、每存活时间成本下降、存活下降不超过 1pp、tracking 恶化不超过 10%。',
            '八组初始化按 seed 配对一致，PPO/HIM 每组各 6000 步，所有参数有限；保存的逐环境 PID 输入可重现最终 PID 状态。',
            '这是固定预算的参数消融，不是 epsilon Pareto frontier，也不是声学降噪证明。两 seed 仅支持机制诊断。', '',
            '完整诊断和单个评估 seed 结果：`legged_gym/logs/quietpaw_gain_20261009/analysis.json`。',
            '实验方案：[quietpaw_gain_protocol_20261009.md](quietpaw_gain_protocol_20261009.md)。']
    text='\n'.join(lines)+'\n'
    (ROOT/'docs/quietpaw_gain_findings_20261009.md').write_text(text)
    print(text)


if __name__=='__main__':
    main()
