import copy, glob, importlib.util, pickle, time
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
for k in ('eval_times','telemetry_vec'): print(k,s[k])
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; A=np.c_[np.ones(len(p['u'])),p['u']]; print(name)
 for k in p:
  if k=='u':continue
  V=p[k]; C=np.linalg.lstsq(A,V,rcond=None)[0]; print(k,'coefs',np.round(C,3),'residual quantiles',np.round(np.quantile(np.linalg.norm(A@C-V,axis=1),[0,.25,.5,.75,.9,1]),5))
print('train',tr['summary'],'traj svd',np.linalg.svd(tr['traj']-tr['traj'].mean(0),compute_uv=False)); print('traj first',tr['traj'][:4]); print('x stats',tr['x'].min(0),tr['x'].max(0))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('candidate',f)); sp.loader.exec_module(m); res=[]; t=time.time()
  for val in [np.arange(52,72),np.arange(0,20),np.random.default_rng(87).choice(72,20,replace=False)]:
   fit=np.setdiff1d(np.arange(72),val); d=copy.copy(s); y=tr['y'][fit]; a=tr['traj'][fit]; d['train']={**{k:tr[k][fit] for k in ('x','y','traj')},'summary':np.array([y.mean(),y.std(),a[:,-1].mean(),a[:,-1].std()])}; d['test']={'x':tr['x'][val]}; o=m.solve_batch(d); e=abs(o['y_hat']-tr['y'][val]); q=abs(o['state_pred']-tr['traj'][val]); res.append(np.round([e.mean(),q.mean(),e.max(),q.mean(axis=0).max()],5).tolist())
  print(f.split('/')[-1],res,round(time.time()-t,2),flush=True)
 except Exception as e: print(f,type(e).__name__,str(e),flush=True)
