import pickle, pathlib
from itertools import combinations
import numpy as np
import final_challenger as model
p=pathlib.Path('/holdout_task/batch_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; x,y,tr=t['x'],t['y'],t['traj']; ix=np.arange(len(x))
splits=[(ix[:52],ix[52:]),(ix[:56],ix[56:])]
for seed in range(10,22):
    z=np.random.default_rng(seed).permutation(ix)
    splits.append((z[:52],z[52:]))
keys=[(d,f) for d in (1,2) for f in ('iso','rot','axes','mixed')]+[(2,'baseline')]
P=[]
for d,f in keys:
    P.append(np.concatenate([model._smooth(x[i],x[j],y[i],d,f) if f!='baseline' else model._gp(x[i],x[j],y[i],1.5,.0001) for i,j in splits]))
Y=np.concatenate([y[j] for _,j in splits]); breaks=np.cumsum([len(j) for _,j in splits])[:-1]
rows=[]
# Compare individuals plus diversified equal-weight pools; do not fit weights to the validation labels.
pools=[(k,) for k in range(len(keys))]+list(combinations(range(8),2))+[tuple(range(4)),tuple(range(4,8)), tuple(range(8)),(0,1,4,5),(0,2,3,4,6,7)]
for ks in pools:
    E=np.array([np.mean(q) for q in np.split(abs(np.mean([P[k] for k in ks],axis=0)-Y),breaks)])
    rows.append(((max(0,E[0]-.038)*2 + E[1]*.15+np.mean(E[2:])+.45*max(E[2:]),len(ks)),ks,E))
for _,ks,E in sorted(rows,key=lambda q:q[0])[:12]:print([keys[k] for k in ks],np.round([E[0],E[1],np.mean(E[2:]),max(E[2:])],6))
_,chosen,E=min(rows,key=lambda q:q[0]); cfg=tuple(keys[k] for k in chosen)
print('Selected 52/20, 56/16, repeat mean, repeat worst',np.round([E[0],E[1],np.mean(E[2:]),max(E[2:])],6))
p.write_text(pathlib.Path('/tmp/final_challenger.py').read_text()+f'\n_READOUT_MODELS = {cfg!r}\n'+'''

def solve_batch(study: dict) -> dict:
    train = study['train']
    x, z = train['x'], study['test']['x']
    t = train['traj']
    mode = np.linalg.svd(t, full_matrices=False)[2][0]
    y = np.mean([_smooth(x, z, train['y'], d, family) if family != 'baseline'
                 else _gp(x, z, train['y'], 1.5, .0001)
                 for d, family in _READOUT_MODELS], axis=0)
    amp = _gp(x, z, t @ mode, 2.4, .001, matern=True)
    state = amp[:, None] * mode
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
''')
