"""Fixed-endpoint E1/E2 findings; no best-checkpoint selection."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'legged_gym/logs/quietpaw_followup_20261009'


def main():
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    manifest=json.loads((OUT/'stability_manifest.json').read_text())
    assert manifest['complete'], 'Do not draw endpoint conclusions from incomplete E2'
    report=dict(replay={},training={})
    lines=['# 热启动稳定性检查：E1 / E2', '',
           '两种初始学习率均采用 fresh Adam、原 adaptive rule、lambda=0；HIM 结构和更新次数不变。', '',
           '## E1：同一 rollout 的首轮更新回放', '',
           '| Seed | 初始 LR | 首次 HIM 更新 KL | 整轮结束 KL |',
           '| --- | ---: | ---: | ---: |']
    for seed in (4,5):
        replay=json.loads((OUT/f'replay{seed}/replay.json').read_text())
        report['replay'][str(seed)]=replay
        for lr,r in replay.items():
            assert r['initial']['kl_mean'] < 1e-6
            lines.append('| {} | {} | {:.6f} | {:.6f} |'.format(
                seed,lr,r['steps'][0]['local']['kl_mean'],r['final']['kl_mean']))
    lines+=['', '## E2：20 轮后随机策略评估', '',
            '评估 seed 111/112，各 4096 个完整首 episode。Cost 下降需与存活率一起解释。', '',
            '| Seed | 初始 LR | 存活率 | Tracking error | 折扣 cost | 每存活秒 cost |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for seed in (4,5):
        pair=[]
        for label,lr in [('high',.001),('low',.00001)]:
            name=f'seed{seed}_{label}'
            directory=OUT/name
            initial=torch.load(directory/'model_him_init.pt',map_location='cpu')
            checkpoint=torch.load(directory/'model_20.pt',map_location='cpu')
            assert checkpoint['iter']==20
            assert all(torch.isfinite(t).all() for t in checkpoint['model_state_dict'].values())
            assert not checkpoint['quietpaw']['pid']['multiplier'].any()
            for key in ['optimizer_state_dict','estimator_optimizer_state_dict']:
                assert not initial[key]['state']
                assert set(int(v['step']) for v in checkpoint[key]['state'].values())=={400}
            pair.append(initial)
            ev=[json.loads((OUT/f'{name}_eval{s}/result.json').read_text()) for s in (111,112)]
            metrics=['survival','tracking','observed_discounted_cost','cost_per_alive_second','alive_seconds']
            values={k:statistics.mean(e['summary'][k] for e in ev) for k in metrics}
            events=EventAccumulator(str(directory)); events.Reload()
            tags=events.Tags()['scalars']
            curves={t:[dict(iteration=v.step,value=v.value) for v in events.Scalars(t)]
                    for t in tags if t in ['Loss/learning_rate','Train/mean_episode_length',
                        'Train/mean_reward','Episode/terrain_level',
                        'QuietPaw/epsilon_bin_0/failure_mean','QuietPaw/epsilon_bin_0/timeout_success_rate']}
            values['curves']=curves
            report['training'][name]=values
            lines.append('| {} | {} | {:.2%} | {:.5f} | {:.6f} | {:.4f} |'.format(
                seed,lr,values['survival'],values['tracking'],values['observed_discounted_cost'],values['cost_per_alive_second']))
        assert all(torch.equal(v,pair[1]['model_state_dict'][k]) for k,v in pair[0]['model_state_dict'].items())
        for key in ['optimizer_state_dict','estimator_optimizer_state_dict']:
            for left,right in zip(pair[0][key]['param_groups'],pair[1][key]['param_groups']):
                assert {k:v for k,v in left.items() if k!='lr'}=={k:v for k,v in right.items() if k!='lr'}
    lines+=['', '四组网络初始化一致（按 seed 配对），初始 Adam 状态均为空；每组 PPO/HIM 均执行 400 次 optimizer.step。', '',
            'E1 定位到大初始 LR 下 HIM 辅助更新引起的策略漂移；E2 只验证短程稳定性。',
            '这不是对 QuietPaw PID 有效性、长期收敛或真实声学降噪的证明。',
            '下一步增益对照必须保持同一稳定初始化，cost/epsilon 单位和 HIM auxiliary objective 不变。']
    (OUT/'findings.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    text='\n'.join(lines)+'\n'
    (ROOT/'docs/quietpaw_followup_findings_20261009.md').write_text(text)
    print(text)


if __name__=='__main__':
    main()
