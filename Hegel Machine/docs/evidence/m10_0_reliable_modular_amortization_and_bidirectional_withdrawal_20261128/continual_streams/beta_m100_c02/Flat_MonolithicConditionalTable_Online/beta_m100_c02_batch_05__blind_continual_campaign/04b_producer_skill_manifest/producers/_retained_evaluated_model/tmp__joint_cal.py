import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as baseline
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X,Y=a['ref_rosette'],a['raw_rosette']; b=a['single_bank_sweeps']; t,v,j=(b[k] for k in ('ref_signal','raw_readout','bank_id'))

def estimate(train,degree):
    # Put isolated and four-channel sweeps on one common scale.
    x=np.r_[t,X.ravel()]; z=np.r_[np.zeros_like(t),np.roll(X,2,axis=1).ravel()]; bank=np.r_[j,np.tile(np.arange(4),len(X))]; obs=np.r_[v,Y.ravel()]; ok=np.r_[np.ones(len(t),bool),np.repeat(train,4)] & (abs(obs)<5)
    def basis(q,k):
        return (chebvander(q/1.8,k)-chebvander(0.,k))[:,1:]
    A=basis(x,9)[:,::2]; B=basis(z,degree)
    def pred(p):
        return p[bank]+np.r_[1.,p[4:7]][bank]*(A@p[7:12]+B@p[12:])
    p0=np.r_[np.zeros(4),1.6,1.,1.6,1.,np.zeros(4+degree)]
    p=least_squares(lambda p:(pred(p)-obs)[ok],p0,loss='soft_l1',f_scale=3e-5,gtol=1e-11,max_nfev=200).x
    return pred(p)[len(t):].reshape(-1,4)-Y
folds=np.array_split(np.random.default_rng(20).permutation(len(X)),8); keep=abs(Y)<5
for degree in (6,8,10,12):
    err=np.zeros_like(Y)
    for idx in folds:
        train=np.ones(len(X),bool); train[idx]=False
        err[idx]=estimate(train,degree)[idx]
    print('joint calibration degree',degree,'OOF median/p95/max',np.quantile(abs(err[keep]),[.5,.95,1]))
