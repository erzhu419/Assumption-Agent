import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,z):return np.linalg.lstsq(x,z,rcond=None)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,k],axis=1) for j in range(d+1) for k in combinations_with_replacement(range(x.shape[1]),j)])
def errs(a):
 z=np.c_[y,T]; r=a@fit(a,z)-z; p=a[52:]@fit(a[:52],z[:52])-z[52:]; return np.round([abs(r[:,0]).mean(),abs(r[:,1:]).mean(),abs(p[:,0]).mean(),abs(p[:,1:]).mean()],5)
np.set_printoptions(precision=4,suppress=True)
b=s['panel_b']; c=s['panel_c']; B=fit(aff(b['u']),b['v']); print('B',B,'sd',np.std(aff(b['u'])@B-b['v'],axis=0)); print('w minmax',w.min(axis=0),w.max(axis=0));print('trajectory singular',np.linalg.svd(T,compute_uv=False))
for label,u in [('C raw',c['u']),('C mapped',aff(c['u'])@B)]:
 print(label,'extrema',u.min(axis=0),u.max(axis=0))
 for k in ('v1','v2'):
  v=c[k]; print(k,'extrema',v.min(axis=0),v.max(axis=0),'affine',fit(aff(u),v),'sd',np.std(aff(u)@fit(aff(u),v)-v,axis=0),'corr',np.corrcoef(u.T,v.T)[:4,4:]); print('sample',np.c_[u[:5],v[:5]])
  for name,fn in [('identity',lambda x:x),('asinh',np.arcsinh),('atan',np.arctan),('tanh',np.tanh),('inv3',lambda x:np.sinh(np.arcsinh(x)/3)),('sin',np.sin),('sinh',np.sinh)]:
   def loss(g):
    a=aff(fn(u*g));return np.mean((a@fit(a,v)-v)**2)
   q=minimize_scalar(loss,bounds=(.05,5),method='bounded'); print(name,round(q.x,4),round(q.fun**.5,6),end='; ')
  print()
for name,u in [('w',w),('calibrated raw',np.linalg.solve(B[1:].T,(w-B[0]).T).T)]:
 for d in (1,2,3): print(name,d,errs(poly(u,d)))
 for k,fn in [('asinh',np.arcsinh),('atan',np.arctan),('tanh',np.tanh),('inv3',lambda x:np.sinh(np.arcsinh(x)/3)),('sin',np.sin),('sinh',np.sinh)]:
  print(name,k,[errs(np.c_[poly(fn(g*u),2),u]) for g in (0.5,1,2,3,5)])
print('archive tails'); print(open('/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py').read().split('def solve_batch')[-1]); print(open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read())
