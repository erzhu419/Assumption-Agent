import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v'))
terms=[()]+[t for k in (1,2,3) for t in combinations_with_replacement(range(4),k)]
X=np.column_stack([np.prod(u[:,t],axis=1) for t in terms]); n=len(u)
A=X[:,:5]; w=np.ones(n)
for _ in range(20):
 c=np.linalg.lstsq(A*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@c-v,axis=1); w=1/np.hypot(1,r/(.7*np.median(r)))
good=np.sort(np.argsort(r)[:-3]); C=np.linalg.lstsq(X[good],v[good],rcond=None)[0]
print('condition of clean full basis',np.linalg.cond(X[good])); np.set_printoptions(precision=8)
for j in range(4):
 print('OUTPUT',j,[(terms[i],round(C[i,j],7)) for i in np.argsort(abs(C[:,j]))[-14:][::-1]])
 keep=np.flatnonzero(abs(C[:,j])>.005); a=X[:,keep]; c=np.linalg.lstsq(a[good],v[good,j],rcond=None)[0]; q=a[good]@c-v[good,j]; h=np.sum(np.linalg.qr(a[good],mode='reduced')[0]**2,axis=1); print('reduced',list(zip([terms[i] for i in keep],c)), 'max',np.max(abs(q)),'LOO max',np.max(abs(q/(1-h))))
print('y / trajectory scales',np.std(s['train']['y']),np.std(s['train']['traj'],axis=0))
