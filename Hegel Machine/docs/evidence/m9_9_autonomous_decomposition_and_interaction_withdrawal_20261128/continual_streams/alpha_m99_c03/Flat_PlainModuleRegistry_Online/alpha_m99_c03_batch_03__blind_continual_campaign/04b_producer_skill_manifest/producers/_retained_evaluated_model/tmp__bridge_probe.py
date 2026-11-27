import pickle
import numpy as np
from itertools import combinations_with_replacement as comb
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; Y=C['bridge_raw']; X=C['true_channel_loads']
def design(x, mode):
    if mode=='diag3': return np.c_[np.ones(len(x)),x,x**3]
    if mode=='ownpoly3': return np.c_[np.ones(len(x)),x,x**2,x**3]
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for d in range(1,4) for ix in comb(range(4),d)])
def fit(A,Y):
    B=np.linalg.lstsq(A,Y,rcond=None)[0]
    for _ in range(35):
        r=np.linalg.norm(A@B-Y,axis=1); w=np.sqrt(np.minimum(1,.002/np.maximum(r,1e-12))); B=np.linalg.lstsq(A*w[:,None],Y*w[:,None],rcond=None)[0]
    return B
for name in ['diag3','ownpoly3','full']:
    A=design(X,name); r=np.empty_like(Y)
    for f in range(5):
        ix=np.arange(len(X))%5==f; r[ix]=A[ix]@fit(A[~ix],Y[~ix])-Y[ix]
    norms=np.linalg.norm(r,axis=1); good=norms<.03
    print(name,'inlier count',sum(good),'median and 90th row residual',np.quantile(norms,[.5,.9]),'inlier RMSE',np.sqrt(np.mean(r[good]**2)),'large rows',[(int(i),float(round(norms[i],4))) for i in np.where(~good)[0]])
    if name=='diag3':
        print('coefficients',fit(A,Y).round(5))
        print('largest residual vectors',r[np.argsort(norms)[-7:]].round(4))
for key in ['train','test']:
    d=D[key]; print(key,{k:getattr(v,'shape',None) for k,v in d.items()}); print('labs',np.unique(d['lab_id'],return_counts=True))
