import pickle
from itertools import combinations_with_replacement as comb
import numpy as np
xdata=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
c=xdata['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
terms=[t for k in range(4) for t in comb(range(4),k)]
def design(x, ts): return np.column_stack([np.prod(x[:,t],axis=1) for t in ts])
def fit(a,y):
 w=np.ones(len(y))
 for _ in range(40):
  b=np.linalg.lstsq(a*w[:,None],y*w[:,None],rcond=None)[0]
  r=np.linalg.norm(a@b-y,axis=1)
  w=1/np.maximum(r, max(1e-7,np.median(r)*2))
 return b
b=fit(design(x,terms),y)
print('coefficients and full residuals')
for t,row in zip(terms,b): print(t,np.round(row,6))
print(np.array2string(np.linalg.norm(design(x,terms)@b-y,axis=1),precision=3))
folds=np.random.default_rng(72).permutation(len(x)).reshape(8,10)
for tag,ts in [('affine',terms[:5]),('quadratic',terms[:15]),('pure-cubic-plus-quadratic',[t for t in terms if len(t)<3 or len(set(t))==1]),('pure-cubics-plus-bilinear',[t for t in terms if len(t)<3 or len(set(t))==1]),('all-cubic',terms),('all-cubic-no-three-way',[t for t in terms if len(set(t))<3])]:
 a=design(x,ts); rr=np.empty(len(x))
 for te in folds:
  tr=np.setdiff1d(np.arange(len(x)),te); rr[te]=np.linalg.norm(a[te]@fit(a[tr],y[tr])-y[te],axis=1)
 clean=rr[~np.isin(np.arange(len(x)),[13,14,15])]
 print(tag,'clean median p95 max',np.quantile(clean,[.5,.95,1]),'worst rows',np.argsort(rr)[-10:])
print('sweeps',[(k,np.asarray(v)[:14]) for k,v in c['single_channel_sweeps'].items()])
