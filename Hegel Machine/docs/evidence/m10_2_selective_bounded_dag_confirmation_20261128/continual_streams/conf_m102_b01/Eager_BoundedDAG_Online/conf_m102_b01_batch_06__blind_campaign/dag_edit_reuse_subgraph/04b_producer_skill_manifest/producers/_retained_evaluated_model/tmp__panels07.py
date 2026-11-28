import numpy as np, pickle, importlib.util
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; y=s['train']['y']; T=s['train']['traj']; n=len(y)
spec=importlib.util.spec_from_file_location('frozen',p+'_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
a,b,c=(s['panel_'+k] for k in 'abc'); raw=s['train']['x']; u=f._inverse(raw,f._fit_panel_a(a['u'],a['v'])); w=f.run_node({'__input__':s})['w'][:n]; print('times telemetry',t,s['telemetry_vec']); print('T singular values',np.linalg.svd(T,compute_uv=False)); print('y vs T coef, rms',np.linalg.lstsq(T,y,rcond=None)[0],np.std(y-T@np.linalg.lstsq(T,y,rcond=None)[0])); print('y std',y.std()); print('feature ranges',[(q.min(axis=0),q.max(axis=0)) for q in (raw,u,w)]);
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ids],axis=1) for k in range(1,d+1) for ids in combinations_with_replacement(range(x.shape[1]),k)])
def fit(x,z): return np.linalg.lstsq(x,z,rcond=None)[0]
for label,q in [('b',b),('c',c)]:
 for key in q:
  if key=='u':continue
  x,z=q['u'],q[key]; print(label,key,'affine coef',fit(poly(x,1),z).round(5));
  for d in (1,2):
   X=poly(x,d); H=X@np.linalg.pinv(X); e=z-H@z; print(d,'rms',np.sqrt(np.mean(e**2,axis=0)),'loo',np.mean(abs(e/(1-np.diag(H)[:,None])),axis=0));
   if d==2: print('quad',fit(X,z).round(5))
for label,x in [('raw',raw),('u',u),('w',w)]:
 for d in (1,2,3):
  X=poly(x,d); z=np.column_stack((y,T)); pred=X[52:]@fit(X[:52],z[:52]); print(label,d,np.mean(abs(pred-z[52:]),axis=0).round(5))
