import numpy as np
from scipy.optimize import nnls
import prony_grid_prior as bench
import prony_rheology as p

def estimate(t,s,y):
    tau=bench.TAU
    duration=np.ptp(t)
    drift=np.c_[np.ones_like(t),(t-t.mean())/duration]
    a=bench.design(t,s,tau)
    base,r=p._calibrate(a,drift,y)
    sig=p._noise(r)
    keep=abs(r-np.median(r))<=4.5*sig
    q=np.linalg.qr(drift[keep],mode='reduced')[0]
    a=a[keep]-q@(q.T@a[keep]); b=y[keep]-q@(q.T@y[keep])
    mag=np.linalg.norm(a@base)/np.linalg.norm(s[keep]-s[keep].mean())
    fast=1/(1+(tau/(2*np.median(np.diff(t))))**2)
    out={'grid':base}
    for slow,ridge in ((12,0),(12,0.5),(12,1),(12,2),(10,0.5),(10,1),(8,1),(8,2)):
        rows=(sig/mag)*np.c_[np.zeros(2),np.stack((slow*tau/duration,fast))]
        # Ridge spreads weakly identified weights across neighboring grid bins
        # instead of requiring one exact relaxation time to carry all the mass.
        extra=(sig/mag)*ridge*np.c_[np.zeros(len(tau)),np.eye(len(tau))]
        aa=np.vstack((a,rows,extra)); rhs=np.r_[b,np.zeros(2+len(tau))]
        norms=np.linalg.norm(aa,axis=0)
        out[f's{slow}_r{ridge:g}']=nnls(aa/norms,rhs,maxiter=1200)[0]/norms
    out['both']=out['s12_r0']
    out['both_mass1.5']=out['s12_r1']
    out['curved16_mass1.5']=out['s8_r2']
    return out,keep

bench.estimate=estimate
bench.run()
