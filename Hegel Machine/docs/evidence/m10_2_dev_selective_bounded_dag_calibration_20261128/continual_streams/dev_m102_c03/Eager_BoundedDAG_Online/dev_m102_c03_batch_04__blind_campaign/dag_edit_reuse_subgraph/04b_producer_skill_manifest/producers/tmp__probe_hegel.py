import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=52
np.set_printoptions(precision=6,suppress=True)
def load(p):
 spec=importlib.util.spec_from_file_location('mod',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('/holdout_task/_frozen_nodes/node_00.py'); a=load('/tmp/hegel_continual_archive/dev_m102_c03_batch_00_solver.py'); sub=dict(s,train={k:v[:n] if k!='summary' else v for k,v in tr.items()},test={'x':x[n:]})
for label,p in [('frozen',m.run_node({'__input__':sub})),('archive',a.solve_batch(sub))]: print(label,np.abs(p['y_hat']-y[n:]).mean(),np.abs(p['state_pred']-T[n:]).mean(axis=0))
print('svd',np.linalg.svd(T,compute_uv=False)); print('diff svd',np.linalg.svd(np.diff(T,axis=1),compute_uv=False));
for kind in ('expm1','exp','damped'):
 for k in (1,2,3,4):
  def basis(r):
   E=np.exp(-t[:,None]*r)
   if kind=='expm1': return 1-E
   if kind=='exp': return np.column_stack([np.ones(len(t)),E])
   if kind=='damped': return np.column_stack([t,1-E])
   return np.column_stack([t]+[t**j*E for j in (1,2)])
  def err(r):
   B=basis(r); return (B@np.linalg.lstsq(B,T.T,rcond=1e-11)[0]-T.T).ravel()
  r=least_squares(err,np.geomspace(.35,2,k),bounds=(.00001,20),max_nfev=2000).x
  print(kind,k,r,np.sqrt(np.mean(err(r)**2)))
B=np.column_stack([np.ones(len(x)),x]); print('linear coefficients',np.linalg.lstsq(B,np.column_stack([y,T[:,0],T[:,-1]]),rcond=None)[0]); print('linear absolute error',abs(B@np.linalg.lstsq(B,np.column_stack([y,T[:,0],T[:,-1]]),rcond=None)[0]-np.column_stack([y,T[:,0],T[:,-1]])).mean(0))
for k in ('panel_a','panel_b','panel_c'):
 u=s[k]['u']; B=np.column_stack([np.ones(len(u)),u]);
 for v in s[k].keys()-{'u'}:
  o=s[k][v]; c=np.linalg.lstsq(B,o,rcond=None)[0]; print(k,v,c,'row residual quantiles',np.quantile(np.linalg.norm(o-B@c,axis=1),[0,.25,.5,.75,.9,1])); print('row paired error quantiles',np.quantile(np.linalg.norm(o-u,axis=1),[0,.25,.5,.75,.9,1]))
