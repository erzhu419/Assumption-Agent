import numpy as np,pickle
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']; np.set_printoptions(precision=4,suppress=True)
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def g(x,degree):
 ix=[p for d in range(degree+1) for p in combinations_with_replacement(range(x.shape[1]),d)]; return np.column_stack([np.prod(x[:,p],axis=1) for p in ix]),ix
for d in [1,2,3]:
 f,ix=g(u,d); ok=np.abs(v)<4
 for j in range(4):
  mask=ok[:,j].copy()
  for i in range(7):
   b=fit(f[mask],v[mask,j]); r=abs(f@b-v[:,j]); mask=ok[:,j]&(r<max(3.5*np.median(r[ok[:,j]]),.0004))
  print('poly',d,j,'med',np.median(r),'kept',mask.sum(),[(p,round(c,4)) for p,c in zip(ix,b) if abs(c)>.008]); print('worst',np.argsort(r)[-9:],np.sort(r)[-9:])
print('train ranges',s['train']['x'].min(0),s['train']['x'].max(0)); print('telemetry',s['telemetry_vec'])
