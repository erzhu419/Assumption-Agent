import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import _coords,_estimate
import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; n=len(a['x']); rng=np.random.default_rng(229);
folds=[np.arange(n),*[rng.permutation(n) for _ in range(8)]]
for b in ['sat','sin','sine']:
 for c in ['c','c4']:
  z,w,q,r=_coords(s,a['x'],b,c)
  for f in [np.sin,np.tanh]:
   errs=[]
   for fold in folds:
    ids,te=fold[:52],fold[52:];p,t=_estimate(z,w,q,r,a['y'],a['traj'],ids,te,fn=f);errs.append((abs(p-a['y'][te]).mean(),abs(t-a['traj'][te]).mean()))
   err=np.asarray(errs);print(b,c,f.__name__,'official',err[0].round(5),'random mean/max',err[1:].mean(0).round(5),err[1:].max(0).round(5),'y folds',err[:,0].round(4),flush=True)
