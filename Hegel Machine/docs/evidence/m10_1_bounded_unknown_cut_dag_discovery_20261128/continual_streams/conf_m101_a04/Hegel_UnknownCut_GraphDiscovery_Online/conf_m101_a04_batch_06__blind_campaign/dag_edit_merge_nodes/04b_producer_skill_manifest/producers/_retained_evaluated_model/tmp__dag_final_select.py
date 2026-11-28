import sys,pickle,types
from pathlib import Path
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00,node_01
base=Path('/tmp/dag_eb3_saved.py').read_text().split("\n_MODE =")[0]; add=Path('/tmp/dag_adaptive_add.py').read_text(); mod=types.ModuleType('model');exec(base+add,mod.__dict__)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y']); rng=np.random.default_rng(224); folds=[]
for i,vi in enumerate([*np.array_split(rng.permutation(n),4),np.arange(n-20,n)]):
    ti=np.setdiff1d(np.arange(n),vi); ss=dict(s); ss['train']={k:s['train'][k][ti] for k in ('x','y','traj')}; ss['test']={'x':s['train']['x'][vi]}
    p=node_00.run_node({'__input__':ss}); b=node_01.run_node({'node_00':p})['b']; folds.append((ss,p,b,np.c_[s['train']['y'][vi],s['train']['traj'][vi]]))
configs=[None,(3,.9,0,0,0),(4,.9,0,0,0),(4,.45,0,0,0),(3,.65,0,0,1),(3,.95,0,0,1),(3,.9,1,0,1),(3,.65,1,0,1),(4,.7,1,0,0),(3,.9,1,0,2),(3,.65,2,0,1),(3,.9,0,4,1),(3,.75,1,4,1),(3,.9,1,5,2),(4,.65,0,4,0),(3,.9,2,0,2),(2,.6,1,0,1),(2,.9,0,4,1),(3,.8,2,4,1),(3,.45,2,4,1)]
preds={}; scores={}
def metrics(ps):
    out=[]
    for pred,(_,_,_,Y) in zip(ps,folds):
        e=abs(pred-Y);out.append([e[:,0].mean(),e[:,1:].mean()])
    return np.array(out)
for q in configs:
    try:
        pp=[mod._fit(ss,p,b,'eb3') if q is None else mod._adaptive(ss,p,b,q) for ss,p,b,_ in folds]; e=metrics(pp)
        preds[q]=pp;scores[q]=e
        print(q,np.round(e,5).tolist(),flush=True)
    except (ValueError,np.linalg.LinAlgError) as err: print(q,str(err),flush=True)
def cvscore(e):return .7*e[:4].mean()+.3*e[:4].max()
top=sorted(preds,key=lambda q:cvscore(scores[q][:,1])); variants=[(q,scores[q][:,1]) for q in top]
for k in (2,3,4,6):
    pick=top[:k]; e=metrics([np.mean([preds[q][i] for q in pick],axis=0) for i in range(len(folds))])[:,1]; variants.append((pick,e));print('ensemble',k,np.round(e,5).tolist(),flush=True)
passing=[v for v in variants if v[1][-1]<=.038]; best,ev=min(passing or variants,key=lambda v:cvscore(v[1])); sy=None
if scores[None][-1,0]>.038:
    sy=min(preds,key=lambda q:cvscore(scores[q][:,0]))
print('chosen',best,ev.tolist(),'scalar',sy,flush=True)
Path('/holdout_task/node_03.py').write_text(base+add+f'\n\n_TRAJ_CONFIG = {best!r}\n_SCALAR_CONFIG = {sy!r}\n')
