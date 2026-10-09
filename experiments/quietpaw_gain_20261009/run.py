"""Sequential, restartable E3 orchestration with frozen-source checks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'legged_gym/logs/quietpaw_gain_20261009'
OLD=ROOT/'experiments/quietpaw_fixed_budget_20261009/worker.py'
HIM=ROOT/'legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_8000.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
    else:
        sources=list((ROOT/'rsl_rl/rsl_rl').rglob('*.py'))+list((ROOT/'legged_gym/legged_gym').rglob('*.py'))
        sources+=list(HERE.glob('*.py'))+list(OLD.parent.glob('*.py'))
        sources+=[ROOT/'docs/quietpaw_gain_protocol_20261009.md']
        assert (OUT/'preflight.json').exists(),'Run preflight first'
        manifest=dict(sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
            him_sha256=sha(HIM),initial_lr=1e-5,epsilon=.037646,seeds=[4,5],gains=[0,1,100,1000],
            iterations=300,num_envs=4096,jobs={},complete=False)

    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(path)

    def check():
        assert sha(HIM)==manifest['him_sha256'],'HIM checkpoint changed'
        for source,expected in manifest['sources'].items():
            assert sha(ROOT/source)==expected,'Frozen source changed: '+source

    def job(name,cmd,artifact):
        check()
        if manifest['jobs'].get(name,{}).get('complete'):
            assert artifact.exists() and sha(artifact)==manifest['jobs'][name]['artifact_sha256']
            return
        if (OUT/name).exists():
            raise RuntimeError('Incomplete/unregistered output exists; preserve and inspect before retry: '+name)
        manifest['jobs'][name]=dict(command=cmd,started=time.time(),complete=False)
        save();print('START',name,flush=True)
        with (OUT/(name+'.log')).open('w') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     env={**os.environ,'WANDB_MODE':'disabled'})
            manifest['jobs'][name]['pid']=process.pid;save()
            while process.poll() is None:
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('RUNNING',name,int(time.time()-manifest['jobs'][name]['started']),flush=True)
            if process.returncode:
                manifest['jobs'][name].update(failed=True,returncode=process.returncode);save()
                raise RuntimeError('Worker failed: '+name)
        assert artifact.exists(),'Missing artifact: '+name
        manifest['jobs'][name].update(complete=True,ended=time.time(),artifact_sha256=sha(artifact))
        save();print('COMPLETE',name,flush=True)

    save();check()
    for seed in manifest['seeds']:
        for gain in manifest['gains']:
            name=f'seed{seed}_g{gain}';folder=OUT/name
            cmd=[sys.executable,'-u',str(HERE/'train.py'),'--seed',str(seed),'--gain',str(gain),'--output',str(folder)]
            if gain:
                cmd+=['--expected-initial',str(OUT/f'seed{seed}_g0/initial.json')]
            job(name,cmd,folder/'model_300.pt')
            for mode,eval_seed in [('evaluate',111),('evaluate',112),('deploy',11)]:
                name2=f'{name}_{mode}{eval_seed}'
                cmd=[sys.executable,'-u',str(OLD),'--mode',mode,'--output',str(OUT/name2),
                    '--him-checkpoint',str(HIM),'--checkpoint',str(folder/'model_300.pt'),
                    '--seed',str(eval_seed),'--epsilon','.037646','--arm','pid']
                job(name2,cmd,OUT/name2/'result.json')
    check();manifest['complete']=True;save();print('TRAINING_AND_EVALUATION_COMPLETE',flush=True)
    subprocess.run([sys.executable,str(HERE/'analyze.py')],cwd=ROOT,check=True)
    manifest['analysis_complete']=True;save();print('STUDY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
