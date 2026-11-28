import pickle, numpy as np
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def poly(x,d):
 terms=[i for k in range(d+1) for i in cwr(range(4),k)]
 return np.column_stack([x[:,i].prod(1) for i in terms]),terms
for pn,vn in [('a','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+pn]; u,v=p['u'],p[vn]; print(pn,vn)
 if pn=='a':
  a,_=poly(u,1); rng=np.random.default_rng(42); w=min((np.linalg.lstsq(a[i],v[i],rcond=None)[0] for i in (rng.choice(len(u),6,replace=False) for _ in range(200))),key=lambda w: np.median(np.linalg.norm(a@w-v,axis=1))); k=np.linalg.norm(a@w-v,axis=1)<2; print('excluded',np.flatnonzero(~k)); u,v=u[k],v[k]
 for degree in ([1,2,3,4] if pn=='a' else [1,2]):
  a,terms=poly(u,degree); w=np.linalg.lstsq(a,v,rcond=None)[0]; e=a@w-v; h=np.sum(np.linalg.qr(a)[0]**2,axis=1); print(degree,np.sqrt(np.mean(e**2)),np.mean(abs(e/np.maximum(1-h[:,None],1e-10))));
  if degree==3 or pn=='c' and degree==2:
   for term,ww in zip(terms,w):
    if max(abs(ww))>.012: print(term,np.round(ww,5))
 print('ranges',np.round(np.c_[u.min(0),u.max(0),v.min(0),v.max(0)],3))
print('train/test ranges',[(k,s[k]['x'].min(0),s[k]['x'].max(0)) for k in ('train','test')])
