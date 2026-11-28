import sys,pickle
import numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _predict,solve_batch,_align,_branch,_ones,_ls
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
a,b=_align(s['panel_a']); z=(x-a)/b; c,d=(_branch(s['panel_c'],z,k) for k in (1,2)); h=-np.expm1(-s['eval_times'][:,None]*[1.8,.4]); L=_ls(h,t.T).T; G=_ones(np.c_[d,c,(d[:,:,None]*c[:,None,:]).reshape(len(z),-1)]); coeff=_ls(G,L); print('mode linear',np.round(coeff[:9],4),'cross',np.round(coeff[9:].T.reshape(2,4,4),4));
for label,ids in [('required',np.arange(52,72))]+[(f'fold{k}',a) for k,a in enumerate(np.array_split(np.random.default_rng(7).permutation(len(x)),4))]+[(f'axis{k}{q}',v) for k in range(4) for q,v in [('low',np.argsort(x[:,k])[:12]),('high',np.argsort(x[:,k])[-12:])]]:
 tr=np.setdiff1d(np.arange(len(x)),ids); p,h=_predict(s,tr,x[ids]); ey=np.abs(p-y[ids]); et=np.abs(h-t[ids]); print(label,'y',ey.mean(),'state',np.round(et.mean(0),5));
 if label=='required': assert ey.mean()<=.038
out=solve_batch(s); assert all(np.isfinite(v).all() for v in out.values()); print({k:v.shape for k,v in out.items()},out['diagnostics_vec'])
