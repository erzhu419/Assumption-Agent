import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares

def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(4),k)])
def fit(A,v,mask): return np.linalg.lstsq(A[mask],v[mask],rcond=None)[0]
def pa(p):
 u,v=p['u'],p['v']; A=poly(u,1); m=np.ones(len(u),bool)
 for _ in range(8):
  r=np.max(abs(A@fit(A,v,m)-v),axis=1);m=r<max(.001,3*np.median(r))
 return m
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a'];m=pa(p); u=p['u'][m];v=p['v'][m]
 for d in [1,2,3,4]:
  A=poly(u,d);b=np.linalg.lstsq(A,v,rcond=None)[0]; E=A@b-v; h=np.sum(np.linalg.svd(A,full_matrices=False)[0]**2,axis=1);print('A',d,np.std(E,0),'loo',np.mean(abs(E/np.maximum(1-h[:,None],1e-10))));
  if d<=3: print(b.round(4).T)
 print('pA raw data moments',u.mean(0),u.std(0),v.mean(0),v.std(0),v.min(0),v.max(0)); print('train x moments',s['train']['x'].mean(0),s['train']['x'].std(0));
 p=s['panel_c'];u=p['u'];
 for k in ['v1','v2']:
  v=p[k]
  for label,A in [('poly2',poly(u,2)),('odd3',np.c_[poly(u,1),poly(u,3)[:,15:]]),('add3',np.c_[poly(u,1),u**2,u**3]),('diag2',np.c_[poly(u,1),u**2]),('coupled2',np.c_[poly(u,1),u*u[:,[1,2,3,0]]])]:
   B=np.linalg.lstsq(A,v,rcond=None)[0];E=A@B-v;h=np.sum(np.linalg.svd(A,full_matrices=False)[0]**2,axis=1);print('C',k,label,np.std(E,0),'loo',np.mean(abs(E/np.maximum(1-h[:,None],1e-10))));
   if label in ('diag2','coupled2'): print(B[5:].round(3).T)
 from importlib.machinery import SourceFileLoader
 mod=SourceFileLoader('prior','/tmp/hegel_continual_archive/conf_m101_b03_batch_00_solver.py').load_module(); H,modes=mod._time_modes(s['eval_times'],s['train']['traj']); print('rates',-np.log1p(-H[0])/s['eval_times'][0], 'trajectory fit MAE',np.mean(abs(modes@H.T-s['train']['traj'])), 'mode correlation',np.corrcoef(np.c_[s['train']['x'],s['train']['y'],modes].T).round(3))
