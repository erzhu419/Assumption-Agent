import pickle, runpy, numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations, combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=135)
a=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); ls=a['_ls']; aff=a['_affine']; C=s['panel_c']; u=C['u']; p,r=a['_panel_operators'](C)
for k in ['panel_a','panel_b','panel_c']:
 print(k,{j: (np.min(z,axis=0),np.max(z,axis=0),np.std(z,axis=0)) for j,z in s[k].items()})
for j,f in enumerate([p,r],1):
 v=C[f'v{j}']; print('V',j,'u/v/residnorm'); print(np.c_[u,v,np.linalg.norm(f(u)-v,axis=1)])
 for f0 in [aff(u),aff(np.column_stack((u,*[u[:,i]*u[:,k] for i,k in combinations_with_replacement(range(4),2)])))]:
  e=v-f0@ls(f0,v); print('polynomial norm',np.linalg.norm(e), 'per col',np.linalg.norm(e,axis=0))
 for name,fn in {'softsign':lambda x,k:x/(1+k*np.abs(x)), 'xrat':lambda x,k:x/(1+k*x*x), 'xsqrat':lambda x,k:x*x/(1+k*x*x), 'absrat':lambda x,k:abs(x)/(1+k*abs(x)), 'sin':lambda x,k:np.sin(k*x), 'cos':lambda x,k:np.cos(k*x), 'tanh':lambda x,k:np.tanh(k*x), 'exp':lambda x,k:np.expm1(-k*x*x), 'xexp':lambda x,k:x*np.exp(-k*x*x)}.items():
  def opt(z):
   def err(k):
    f=aff(np.c_[z,fn(z,k)]); return np.linalg.norm(v-f@ls(f,v))
   res=min((minimize_scalar(err,bounds=(l,h),method='bounded') for l,h in [(.001,.3),(.3,1.1),(1.1,2.4),(2.4,6),(6,12)]),key=lambda t:t.fun); return (res.fun,res.x)
  b,m=a['_calibration'](s['panel_b']); d,A=a['_calibration'](s['panel_a']); print(name,[(round(e,5),round(k,4)) for e,k in (opt(z) for z in [u,u@m+b,np.linalg.solve(m.T,(u-b).T).T,u@A+d])])
for k in ['panel_a','panel_b']:
 b,m=a['_calibration'](s[k]); print(k,'matrix sv',np.linalg.svd(m,compute_uv=False));
frozen=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py'); tr=s['train']['traj']; B=frozen['_temporal_basis'](s['eval_times'],tr[:52]); print('frozen temporal irreducible MAE',abs(tr[52:]-(tr[52:]@B)@B.T).mean()); print('traj means',tr.mean(0)); print('x quantiles',np.quantile(s['train']['x'],[0,.1,.5,.9,1],axis=0));
