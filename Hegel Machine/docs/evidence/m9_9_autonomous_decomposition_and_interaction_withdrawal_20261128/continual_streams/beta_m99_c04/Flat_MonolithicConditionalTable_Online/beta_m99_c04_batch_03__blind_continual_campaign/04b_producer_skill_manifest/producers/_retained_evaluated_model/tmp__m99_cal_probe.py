import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
terms=[()] + [(i,) for i in range(4)] + list(combinations_with_replacement(range(4),2)) + list(combinations_with_replacement(range(4),3))
A=np.column_stack([x[:,t].prod(1) for t in terms]); keep=np.arange(len(x)); folds=np.array_split(np.random.default_rng(718).permutation(keep),8)
def fit(A,y):
    # Use low residual consensus before estimating the near-noise-scale model.
    sel=np.arange(len(y)); b=np.linalg.lstsq(A,y,rcond=None)[0]
    for _ in range(7):
        err=np.linalg.norm(y-A@b,axis=1)
        sel=np.argsort(err)[:int(np.ceil(.84*len(y)))]; b=np.linalg.lstsq(A[sel],y[sel],rcond=None)[0]
    err=np.linalg.norm(y-A@b,axis=1); sel=err < max(.04,4*np.median(err)); b=np.linalg.lstsq(A[sel],y[sel],rcond=None)[0]
    return b
for n in [5,15,35]:
    b=fit(A[:,:n],y); r=y-A[:,:n]@b; cv=np.zeros_like(r)
    for t in folds:
        tr=np.setdiff1d(keep,t); cv[t]=y[t]-A[t,:n]@fit(A[tr,:n],y[tr])
    good=np.linalg.norm(cv,axis=1)<3
    print('degree columns',n,'bad rows',np.where(~good)[0],'cv quantiles clean',np.quantile(np.abs(cv[good]),[.5,.90,.99,1]),'residual clean',np.quantile(np.abs(r[good]),[.5,.9,.99,1]))
    if n<=15: print(list(zip(terms,b.round(5).tolist())))
    else:
        for j in range(4): print(j,[(terms[i],round(b[i,j],5)) for i in range(n) if abs(b[i,j])>.008])
print('outlier rows ref/raw',np.c_[x[12:17],y[12:17]].round(3))
