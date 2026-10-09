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
    lines += ['', '## 判断与后续实验', '',
        '首要已证实问题是热启动参数与训练流程的匹配，不是已经证实的 cost / GAE / PID 公式错误。',
        '原 HIM8000 的 PPO/HIM optimizer 保存 LR 均为 1e-5；weight-only warm-start 清空 Adam 状态后却从 1e-3 开始。',
        '第一 minibatch 的 adaptive rule 将其升至 1.5e-3，随后的 HIM 更新显著改变 actor 输入；PPO 梯度 detach 不能阻止这种输入分布变化。',
        '同一 rollout 回放中，改为初始 1e-5 即将首次 HIM 更新后的策略 KL 从约 14 降到约 0.0016，随后短程行走性能也明显恢复。', '',
        '当前仅在独立实验入口设置初始 LR；生产训练默认值、HIM 网络/损失/更新次数、cost 和 PID 公式均未改。',
        '高 LR 的 seed5 在失败增加时出现更低 cost，不能据此认为它比低 LR 更好。', '',
        '后续 E3 已预先设计但尚未运行：两 seed × PID gain scale {0,1,100,1000}，统一初始 LR=1e-5、fresh Adam、epsilon=0.037646，固定 300 轮。',
        '首先解释 scale=0/1 的同预算比较；100/1000 只检验参数尺度，不作为新优化方法或作者公开超参数。',
        '保留原 adaptive schedule，记录每 minibatch 的 LR/KL、HIM 漂移、实际 reward/cost 梯度和失败率；若出现再次失稳，单独检验 LR cap，不偷偷加入。',
        '训练后在固定 checkpoint 上同时判断预算、存活、tracking 与每存活时间 cost；禁止按中途最优点选结果。',
        '若较大有效成本梯度仍不能降低 cost，再检查 critic/advantage、触地采样口径和预算可行性；暂不加入 Stage 2。', '',
        '数值尺度对 PID-Lagrangian 调参的重要性也见 [Stooke et al. 2020](https://proceedings.mlr.press/v119/stooke20a.html)。这里不引入该文额外的尺度不变优化机制。', '',
        '完整协议：[quietpaw_followup_protocol_20261009.md](quietpaw_followup_protocol_20261009.md)。',
        '完整数据：`legged_gym/logs/quietpaw_followup_20261009/findings.json`；命令和源码哈希见同目录 `stability_manifest.json`。',
        '环境：现有 conda himloco；36 项 QuietPaw 测试执行通过；所有四组 checkpoint 参数有限，lambda=0，PPO/HIM 各 400 步。']
    (OUT/'findings.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    text='\n'.join(lines)+'\n'
    (ROOT/'docs/quietpaw_followup_findings_20261009.md').write_text(text)
    print(text)


if __name__=='__main__':
    main()
