"""Run the fixed E2 matrix after both E1 replay conditions pass."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'legged_gym/logs/quietpaw_followup_20261009'
HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/quietpaw_fixed_budget_20261009/worker.py'
HIM=ROOT/'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for seed in (4,5):
        replay=json.loads((OUT/f'replay{seed}/replay.json').read_text())
        assert replay['1e-05']['final']['kl_mean'] < replay['0.001']['final']['kl_mean']
    path=OUT/'stability_manifest.json'
    if path.exists():
        raise RuntimeError('Study already registered; inspect it before any restart')
    sources=list((ROOT/'rsl_rl/rsl_rl').rglob('*.py'))+list((ROOT/'legged_gym/legged_gym').rglob('*.py'))
    sources+=list(HERE.glob('*.py'))+list(OLD.parent.glob('*.py'))
    sources+=[ROOT/'docs/quietpaw_followup_protocol_20261009.md']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    manifest=dict(sources=hashes,him_sha256=sha(HIM),jobs={},complete=False)

    def save():
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest,indent=2))
        tmp.replace(path)

    def job(name,command):
        for source,expected in hashes.items():
            assert sha(ROOT/source)==expected,source
        assert sha(HIM)==manifest['him_sha256']
        manifest['jobs'][name]=dict(command=command,started=time.time(),complete=False)
        save()
        print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     env={**os.environ,'WANDB_MODE':'disabled'})
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('RUNNING',name,flush=True)
            if process.returncode:
                raise RuntimeError('Failed: '+name)
        manifest['jobs'][name].update(complete=True,ended=time.time())
        save()
        print('COMPLETE',name,flush=True)

    save()
    for seed in (4,5):
        initial=None
        for label,lr in [('high',.001),('low',.00001)]:
            name=f'seed{seed}_{label}'
            directory=OUT/name
            job(name,[sys.executable,'-u',str(HERE/'stability.py'),'--seed',str(seed),
                      '--lr',str(lr),'--output',str(directory)])
            current=json.loads((directory/'initial.json').read_text())
            # Optimizer configuration intentionally differs in LR only; state
            # dictionaries are empty at initialization in both branches.
            common={k:v for k,v in current.items() if k!='optimizer'}
            if initial is not None:
                assert initial==common,'Paired model/RNG/environment mismatch'
            initial=common
            for eval_seed in (111,112):
                eval_name=f'{name}_eval{eval_seed}'
                job(eval_name,[sys.executable,'-u',str(OLD),'--mode','evaluate','--output',str(OUT/eval_name),
                    '--him-checkpoint',str(HIM),'--checkpoint',str(directory/'model_20.pt'),
                    '--seed',str(eval_seed),'--epsilon','.037646','--arm','zero'])
    manifest['complete']=True
    save()
    print('STABILITY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
