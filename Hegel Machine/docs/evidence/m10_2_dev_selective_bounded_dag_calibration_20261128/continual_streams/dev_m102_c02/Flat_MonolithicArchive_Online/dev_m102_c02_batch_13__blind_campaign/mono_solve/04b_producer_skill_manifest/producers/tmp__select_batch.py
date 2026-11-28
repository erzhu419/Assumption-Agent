import pickle,sys
import numpy as np
import batch_base as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];b,u=m._features(s,tr['x']);h=m.aff(b)
n=len(b); rng=np.random.default_rng(87)
splits=[np.arange(52,72),np.arange(20),np.arange(26,46)]+[rng.choice(n,20,replace=False) for _ in range(14)]
errs={}; trajectory=[]; horizons=[]
for val in splits:
    idx=np.setdiff1d(np.arange(n),val); pp=m._predictions(b,u,tr,idx)
    for k,p in pp.items():errs.setdefault(k,[]).append(float(np.mean(abs(p[val]-tr['y'][val]))))
    e=np.abs(h[val]@m.fit(h[idx],tr['traj'][idx])-tr['traj'][val]);trajectory.append(e.mean());horizons.append(e.mean(axis=0).max())
def score(k):
    e=errs[k][1:];return .7*np.mean(e)+.3*np.max(e)+(0 if isinstance(k,str) else .0006)
order=sorted(errs,key=score)
print('best stress candidates:',[(k,round(errs[k][0],6),round(np.mean(errs[k][1:]),6),round(max(errs[k][1:]),6)) for k in order[:14]])
# Prefer the best repeated-split performer among candidates meeting the public gate.
k=next((k for k in order if errs[k][0]<=.0378),('u',3,1.,0,.3))
print('selected:',k,'public scalar, trajectory:',errs[k][0],trajectory[0],'stress mean scalar:',np.mean(errs[k][1:]),'worst scalar, trajectory, horizon:',max(errs[k]),max(trajectory),max(horizons))
open('/holdout_task/batch_solver.py','w').write(open('/tmp/batch_base.py').read()+f'\n\n_SCALAR_MODEL = {k!r}\n')
