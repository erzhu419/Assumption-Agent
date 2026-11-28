import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['y']); allrows=np.arange(N)
def check(name,v):
    a=np.setdiff1d(allrows,v); d=dict(s); d['train']={**tr,**{k:tr[k][a] for k in ('x','y','traj')}}; d['test']={'x':tr['x'][v]}
    p=solve_batch(d); ey=np.abs(p['y_hat']-tr['y'][v]); et=np.abs(p['state_pred']-tr['traj'][v]); print(name,round(ey.mean(),6),round(et.mean(),6),np.round(et.mean(0),6),round(float(np.max(et.mean(0))),6)); assert max(ey.mean(),et.mean(),et.mean(0).max()) <= .038
check('52/20',allrows[52:]); rng=np.random.default_rng(1323)
for j,v in enumerate(np.array_split(rng.permutation(N),4)):check('OOF '+str(j),v)
for j in range(4):
    ix=np.argsort(tr['x'][:,j]); check('low '+str(j),ix[:16]); check('high '+str(j),ix[-16:])
check('large norm',np.argsort(np.linalg.norm(tr['x'],axis=1))[-16:])
