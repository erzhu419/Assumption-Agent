import sys
sys.path.insert(0, '/holdout_task')
from batch_solver import _features, _predict
import numpy as np
import pickle
from itertools import product
from pathlib import Path

tr=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['train']
x,y,traj=tr['x'],tr['y'],tr['traj']; ix=np.arange(len(y))
splits=[(ix[:52],ix[52:])]
for seed,n,k in [(36,52,5),(1124,56,8),(971,56,8)]:
    rng=np.random.default_rng(seed)
    for _ in range(k):
        p=rng.permutation(ix); splits.append((p[:n],p[n:]))
fs=[(_features(x[a],x[b],y[a],traj[a]), y[a], y[b]) for a,b in splits]
ends=np.cumsum([0]+[len(g) for f,t,g in fs]); lens=np.diff(ends)

def metrics(e):
    return np.add.reduceat(abs(e),ends[:-1],axis=-1)/lens

cfgs=set()
for metric,warp,latent,use in product(((0.,0.),(1.,.4),(1.,.8),(1.,1.5)),(-.075,0.,.075),(0.,.30,.55),(0,1)):
    if use and latent: continue
    for length,nugget in product((.70,.76,.82,.89,.96,1.04,1.12,1.22,1.34),(3e-6,3e-5,1e-4,4e-4)):
        cfgs.add((*metric,warp,latent,use,length,nugget))
cfgs=sorted(cfgs)
err=np.array([np.concatenate([_predict(f,t,cfg)-g for f,t,g in fs]) for cfg in cfgs]); m=metrics(err)
indices=np.arange(len(cfgs)); avg=m.mean(1); worst=m.max(1)
short=np.unique(np.r_[np.argsort(m[:,0])[:42],np.argsort(.46*avg+.39*worst+.15*m[:,0])[:75],np.argsort(worst)[:24],np.argsort(avg)[:24]])
# The published split is a feasibility constraint, not the objective among feasible models.
def objective(sc):
    return .52*sc.mean(axis=-1)+.48*sc.max(axis=-1)+5*np.maximum(sc[...,0]-.03785,0)

best=[]
for i in short:
    for w in (.25,.4,.5,.6,.75,1.):
        e=w*err[i]+(1-w)*err[short]; sc=metrics(e); s=objective(sc)
        j=s.argmin(); best.append((s[j], ((w,i),(1-w,short[j])), e[j], sc[j]))
best.sort(key=lambda r:r[0]); top=best[0]
for _, models, e, _ in best[:24]:
    for w in (.125,.25,.375):
        combined=(1-w)*e+w*err; sc=metrics(combined); s=objective(sc); j=s.argmin()
        if s[j]<top[0]: top=(s[j],tuple(((1-w)*v,i) for v,i in models)+((w,j),),combined[j],sc[j])
_, models, _, sc=top
cfg=[(w,cfgs[i]) for w,i in models if w]
print('Selected:',cfg,'target required/mean/worst:',sc[0],sc.mean(),sc.max(), 'extra 56/16 mean/worst:',sc[-8:].mean(),sc[-8:].max())
p=Path('/holdout_task/batch_solver.py')
with p.open('a') as out: out.write('\n\n_SCALAR_MODELS = '+repr(cfg)+'\n')
