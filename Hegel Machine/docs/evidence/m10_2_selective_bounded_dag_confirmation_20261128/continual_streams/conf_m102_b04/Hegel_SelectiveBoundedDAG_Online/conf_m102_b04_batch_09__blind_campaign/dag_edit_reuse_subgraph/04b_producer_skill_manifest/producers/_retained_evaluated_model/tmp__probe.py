import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=run_node({'__input__':s}); x=np.cbrt(np.vstack([r['cubic_train'],r['cubic_test']])); y=s['train']['y']; t=s['train']['traj']; print('ranges',x.min(0),x.max(0)); print('traj sv',np.linalg.svd(t,compute_uv=False));print('time',s['eval_times'],'telemetry',s['telemetry_vec']);
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
for p in ('panel_b','panel_c'):
 u=s[p]['u'];print(p)
 for k,v in s[p].items():
  if k=='u':continue
  for desc,q in [('u',u),('cubic',u**3),('u+cubic',np.c_[u,u**3]),('atanh',u),('v1',s[p].get('v1',u))]:
   if desc=='atanh':
    if abs(v).max()>=1:continue
    z=np.arctanh(v)
   else:z=v
   for d in ([1,2] if desc in ('u','v1') else [1]):
    f=poly(q,d); a=np.linalg.pinv(f); e=z-f@(a@z); h=np.einsum('ij,ji->i',f,a);print(k,desc,d,np.round(np.mean(abs(e)/np.maximum(1-h[:,None],1e-9),0),5))
  print('v range',v.min(0),v.max(0)); print('affine fit',np.round(np.linalg.lstsq(poly(u,1),v,rcond=None)[0],3))
print('scalar ranges',y.min(),y.max()); print('trajectory normalized',np.round((t/np.maximum(np.abs(t[:,-1,None]),.1))[:10],3))
