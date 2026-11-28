import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def _self_basis(u,b,kind):
    q=u*u
    if kind=='exp': return u*np.exp(-b[0]*q)
    if kind=='tanh': return np.tanh(b[0]*u)
    if kind=='erf': return erf(b[0]*u)
    if kind=='rational': return u/(1+b[0]*q)
    if kind=='power': return u/(1+b[0]*q)**b[1]
    if kind=='quartic': return u/(1+b[0]*q+b[1]*q*q)
    if kind=='logistic': return u/(1+np.exp(np.clip(b[0]*(q-b[1]),-50,50)))

def _estimate(sweeps, kind='exp', fold=None, extra=False):
    x=sweeps['ref_rosette']; y=sweeps['raw_rosette']; single=sweeps['single_bank_sweeps']; ids=np.ravel(single['bank_id']); sig=np.ravel(single['ref_signal']); raw=np.ravel(single['raw_readout']); ds=[]
    for k in range(4):
        ix=ids==k; m=np.zeros((sum(ix),4)); m[:,k]=sig[ix]
        xx=np.r_[x,m]; yy=np.r_[y[:,k],raw[ix]]; idx=np.arange(len(yy)); good=np.abs(yy)<5
        ds.append((xx,yy,good,idx))
    def design(x,k,b):
        u=x[:,k]; v=x[:,(k+2)%4]; l=x[:,(k+1)%4]; r=x[:,(k+3)%4]; cols=[np.ones(len(x)),_self_basis(u,b[:-1],kind),u,u**3,v,v*v/(1+b[-1]*v*v)]
        if extra: cols.extend((l,r,u*l,u*r,l*r))
        return np.column_stack(cols)
    masks=[g if fold is None else g&(idx%4!=fold) for _,_,g,idx in ds]
    def profile(b, ret=False):
        cs=[]; rs=[]
        for k,((x,y,_,_),tr) in enumerate(zip(ds,masks)):
            A=design(x[tr],k,b); t=y[tr]; c=np.linalg.lstsq(A,t,rcond=1e-11)[0]
            for _ in range(7):
                w=(1+((A@c-t)/.001)**2)**(-.25); c=np.linalg.lstsq(A*w[:,None],t*w,rcond=1e-11)[0]
            cs.append(c); r=A@c-t; rs.append(r*np.sqrt(2/(1+np.sqrt(1+(r/.001)**2))))
        return cs if ret else np.concatenate(rs)
    b0=np.array([.6,.7] if kind not in ('power','quartic','logistic') else [.6,1.,.7]); opt=least_squares(profile,b0,bounds=(.001,30.),max_nfev=100,diff_step=1e-4); b=opt.x; cs=profile(b,True)
    def predict(x): return np.column_stack([design(x,k,b)@c for k,c in enumerate(cs)])
    errors=[design(x[g if fold is None else g&(idx%4==fold)],k,b)@cs[k]-y[g if fold is None else g&(idx%4==fold)] for k,(x,y,g,idx) in enumerate(ds)]
    return predict,errors,b,cs

def _calibrate(study):
    predict,_,__,___=_estimate(study['sensor_calibration_sweeps']); raw=np.asarray(study['y_obs_raw']).reshape(-1,4)
    return np.concatenate([least_squares(lambda u:predict(u[None,:])[0]-y,np.zeros(4),ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=80).x for y in raw])
