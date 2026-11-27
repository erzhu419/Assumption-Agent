import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/m98_arrays.npz'); globals().update({k:z[k] for k in z.files})
drift=th[2]+x@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
G=(hold[:,None]-past@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)))*th[:2]

def predict(p, mode):
    h,gain,logtime = p[:3]
    fast=p[3] if mode=='free' else 0.
    T=tau*np.exp([fast,h]); s=np.exp(logtime); d=np.exp(gain)*drift
    if mode=='quadratic': d=d+p[3]*drift**2
    return G@(-np.expm1(-t[None,:]/T[:,None]))-d[:,None]*s*np.expm1(-t/s), G@(1/T)+d

def metrics(c,r):
    return np.round([np.median(abs(c)),np.median(abs(r)),max(np.median(abs(c[labs==l])) for l in np.unique(labs)),max(np.median(abs(r[labs==l])) for l in np.unique(labs)),max(np.median(abs(c),axis=0))],5)
for mode in ['fixed_fast','free','quadratic']:
    pc=np.zeros_like(yc); pr=np.zeros_like(yr)
    def fit(mask):
        def resid(p):
            c,r=predict(p,mode)
            return np.r_[((c[mask]-yc[mask])/.02).ravel(),(r[mask]-yr[mask])/.05]
        return least_squares(resid,np.zeros(3+(mode!='fixed_fast')),loss='soft_l1',f_scale=1.5).x
    for lab in np.unique(labs):
        m=labs==lab; f,r=predict(fit(~m),mode); pc[m]=f[m]; pr[m]=r[m]
    p=fit(np.ones(len(labs),dtype=bool))
    print(mode,'LOLO',metrics(pc-yc,pr-yr),'params',p,'longest-time errors',np.quantile(abs(pc[:,-1]-yc[:,-1]),[.5,.9,1]))
