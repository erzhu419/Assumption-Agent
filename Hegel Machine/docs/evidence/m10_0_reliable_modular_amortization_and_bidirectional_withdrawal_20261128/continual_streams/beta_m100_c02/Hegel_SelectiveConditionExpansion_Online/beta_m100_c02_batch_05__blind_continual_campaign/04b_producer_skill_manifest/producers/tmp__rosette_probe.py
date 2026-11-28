import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; n=len(x)
np.set_printoptions(precision=6,suppress=True)
for k,v in c['single_bank_sweeps'].items(): print(k,np.shape(v),np.asarray(v).ravel()[:16])
def design(z,key):
    a=[np.ones(len(z)),z]; t=z*z
    terms={'affine':[], 'square':[t], 'cube':[z*t], 'diagpoly':[t,z*t], 'poly2':[np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])], 'tanh':[np.tanh(z)-z], 'rat3':[z*t/(1+t)], 'rat2':[t/(1+t)], 'sin':[np.sin(z)-z], 'softsign':[z/(1+abs(z))-z], 'absquad':[z*abs(z)]}
    return np.column_stack(a+terms[key])
def robust(a,b):
    m=np.linalg.lstsq(a,b,rcond=None)[0]
    for scale in [0.3,.1,.03,.01,.003,.001]:
        for _ in range(15):
            r=np.linalg.norm(a@m-b,axis=1); w=1/np.hypot(1,r/scale); m=np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]
    return m
base=design(x,'affine'); m=robust(base,y); r=y-base@m; rr=np.linalg.norm(r,axis=1); print('affine',m,'res quantiles',np.quantile(rr,[0,.5,.8,.9,.95,1])); print('largest',np.sort(rr)[-14:]); print('example clean x,y,res',np.c_[x,y,r][np.argsort(rr)[[5,10,20,30,40,50,60,68,69,70]]]); print('ref correlation',np.corrcoef(x.T)); print('raw obs quantiles',np.quantile(s['y_obs_raw'].reshape(-1,4),[0,.5,1],axis=0)); print('rows',s['row_bank_id'][:32]); folds=np.random.default_rng(20).permutation(n)%5
for key in ['affine','square','cube','diagpoly','poly2','tanh','rat3','rat2','sin','softsign','absquad']:
    a=design(x,key); o=np.zeros(n); m=robust(a,y)
    for j in range(5):
        v=folds==j; o[v]=np.linalg.norm(a[v]@robust(a[~v],y[~v])-y[v],axis=1)
    print(key,np.quantile(o,[.25,.5,.75,.85,.9]),'clean fold max med',max(np.median(o[(folds==j)&(rr<1)]) for j in range(5))); print(m if key not in ['affine','poly2','diagpoly'] else '')
