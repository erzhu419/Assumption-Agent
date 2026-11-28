import pickle,numpy as np
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; L=np.linalg.lstsq
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aff(x):return np.c_[np.ones(len(x)),x]
def mon(x,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(x[:,idx],axis=1) for k in range(d+1) for idx in combinations_with_replacement(range(x.shape[1]),k)])
def robust(H,V):
 keep=np.ones(len(H),bool)
 for i in range(12):
  w=L(H[keep],V[keep],rcond=None)[0]; err=np.linalg.norm(H@w-V,axis=1); k=err<max(5*np.median(err),1e-4)
  if np.array_equal(k,keep) or sum(k)<H.shape[1]:break
  keep=k
 print('in',sum(keep),'res',np.quantile(err,[.1,.5,.8,.9,1]));return w
if __name__=='__main__':
 print('telemetry',s['telemetry_vec'])
 for p in ('panel_a','panel_b','panel_c'):
  u=s[p]['u']; print(p)
  for k,v in s[p].items():
   if k=='u':continue
   for d in (1,2,3):
    H=mon(u,d);print(k,d,end=' ')
    if p=='panel_a':
     w=robust(H,v)
     if d<=2:print(w)
    elif d<=2:
     w=L(H,v,rcond=None)[0]; print(np.sqrt(np.mean((H@w-v)**2)), np.sqrt(np.mean(((H@w-v)/(1-np.einsum('ij,ji->i',H,np.linalg.pinv(H)))[:,None])**2))); print(w if d==1 else '')
    else: print()
 t=s['train']['traj'];print('traj svd',np.linalg.svd(t,compute_uv=False)); print('centered',np.linalg.svd(t-t.mean(0),compute_uv=False))
