import pickle,numpy as np
from itertools import combinations_with_replacement
from collections import Counter
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; b=C['bridge_raw']; x=C['true_channel_loads']; print('shapes',x.shape,b.shape,'nonzero counts',Counter(np.count_nonzero(x,axis=1))); print('first rows',np.round(np.c_[x,b][:17],4)); print('unique load',*[len(np.unique(x[:,j])) for j in range(4)])
rng=np.random.default_rng(602);fold=np.empty(len(x),int);fold[rng.permutation(len(x))]=np.arange(len(x))%5

def terms(z, degree, full):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,ix],axis=1) for j in range(1,degree+1) for ix in combinations_with_replacement(range(4),j) if full or len(set(ix))==1])
def assess(name,a,y):
 pred=np.zeros_like(y)
 for f in range(5):
  tr=fold!=f;pred[~tr]=a[~tr]@np.linalg.lstsq(a[tr],y[tr],rcond=None)[0]
 err=pred-y; print(name,'rmse/max',np.sqrt(np.mean(err**2)),abs(err).max(),'nonzero slices',[(n,round(np.sqrt(np.mean(err[np.count_nonzero(x,axis=1)==n]**2)),6)) for n in np.unique(np.count_nonzero(x,axis=1))]); return np.linalg.lstsq(a,y,rcond=None)[0]
for z,y,label in [(x,b,'forward'),(b,x,'inverse')]:
 for d in (1,2,3,4,5):
  for full in ([False,True] if d in (2,3) else [False]):
   a=terms(z,d,full); c=assess((label,d,full),a,y)
   if label=='forward' and d==3: print('terms coefficient norms',np.round(np.linalg.norm(c,axis=1),5));print('pure coeff' if not full else 'full cubic max',np.round(c,4) if not full else np.max(abs(c)))
