import itertools, pickle
import numpy as np
xdata=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']
x,y=xdata['ref_rosette'],xdata['raw_rosette']
base=[()]+[(i,) for i in range(4)]
quad=list(itertools.combinations_with_replacement(range(4),2))
cube=[(i,)*3 for i in range(4)]
terms={'linear':base,'quadratic':base+quad,'compact':base+[t for t in quad if len(set(t))==2]+cube,'compact_sq':base+quad+cube,'full':base+quad+list(itertools.combinations_with_replacement(range(4),3))}
terms['fifth']=terms['compact_sq']+[(i,)*d for d in (4,5) for i in range(4)]
def basis(x,t): return np.column_stack([np.prod(x[:,q],axis=1) for q in t])
def fit(A,B):
    b=np.linalg.lstsq(A,B,rcond=None)[0]
    for _ in range(24):
        r=np.linalg.norm(B-A@b,axis=1)
        s=max(1e-10,1.5*np.median(r)); w=np.sqrt(np.minimum(1,s/np.maximum(r,1e-30)))
        b=np.linalg.lstsq(A*w[:,None],B*w[:,None],rcond=None)[0]
    r=np.linalg.norm(B-A@b,axis=1)
    keep=r<=max(8*np.median(r),1e-8)
    if keep.sum()>=A.shape[1]: b=np.linalg.lstsq(A[keep],B[keep],rcond=None)[0]
    return b,keep
folds=np.array_split(np.arange(len(x)),5)
for name,t in terms.items():
    A=basis(x,t); e=np.zeros(len(x)); flagged=[]
    for test in folds:
        train=np.setdiff1d(np.arange(len(x)),test); b,keep=fit(A[train],y[train]); flagged.append(train[~keep].tolist()); e[test]=np.max(abs(A[test]@b-y[test]),axis=1)
    good=np.abs(y).max(axis=1)<5
    b,keep=fit(A,y)
    print(name,'max',round(max(e[good]),8),'median',round(np.median(e[good]),8),'folds',[round(max(e[z][good[z]]),8) for z in folds],'train flagged',np.flatnonzero(~keep).tolist(),'oof-fit flagged',flagged)
    if name=='compact': print('linear',np.round(b[1:5],5),'min_sv',np.linalg.svd(b[1:5],compute_uv=False)[-1])
