import sys,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01
from dag_sink_challenger import _fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y'])
rng=np.random.default_rng(224)
folds=[]
for vi in [*np.array_split(rng.permutation(n),4),np.arange(n-20,n)]:
    ti=np.setdiff1d(np.arange(n),vi); ss=dict(s); ss['train']={k:s['train'][k][ti] for k in ('x','y','traj')}; ss['test']={'x':s['train']['x'][vi]}
    p=node_00.run_node({'__input__':ss}); b=node_01.run_node({'node_00':p})['b']; folds.append((ss,p,b,np.c_[s['train']['y'][vi],s['train']['traj'][vi]]))
results={}
for mode in ('base','ridge','eb3','eb4'):
    e=[]
    for ss,p,b,z in folds:
        err=abs(_fit(ss,p,b,mode)-z);e.append([err[:,0].mean(),err[:,1:].mean()])
    results[mode]=np.array(e);print(mode,np.round(e,5).tolist(),flush=True)
passed=[m for m,e in results.items() if e[-1].max()<=.038]
best=min(passed or list(results),key=lambda m:results[m][:-1].mean())
print('selected',best,flush=True)
if best!='base':
    a=Path('/tmp/dag_sink_challenger.py').read_text().replace("_MODE = 'eb4'",f'_MODE = {best!r}')
    Path('/holdout_task/node_03.py').write_text(a)
