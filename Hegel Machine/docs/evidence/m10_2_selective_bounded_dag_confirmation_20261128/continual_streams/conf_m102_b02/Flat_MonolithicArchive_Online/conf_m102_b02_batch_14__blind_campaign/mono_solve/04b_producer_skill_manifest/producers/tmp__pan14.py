import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for k in ('a','b','c'):
 d=s['panel_'+k];u=d['u']; U=np.column_stack([np.prod(u[:,ix],axis=1) for p in range(4) for ix in combinations_with_replacement(range(4),p)])
 for key,v in d.items():
  if key=='u':continue
  for n in [5,9,15,19,23,35]:
   D=U[:,:n]; w=np.ones(len(u));
   if k=='a':
    for _ in range(25):
     c=np.linalg.lstsq(w[:,None]*D,w[:,None]*v,rcond=None)[0];r=np.linalg.norm(D@c-v,axis=1);w=np.minimum(1,.04/np.maximum(r,1e-10))**.8
   c=np.linalg.lstsq(w[:,None]*D,w[:,None]*v,rcond=None)[0];r=np.linalg.norm(D@c-v,axis=1); print(k,key,n,np.round(np.quantile(r,[.1,.5,.8,.9,1]),5));
   if n in ([35] if k=='a' else [15]):print(np.round(c,3))
