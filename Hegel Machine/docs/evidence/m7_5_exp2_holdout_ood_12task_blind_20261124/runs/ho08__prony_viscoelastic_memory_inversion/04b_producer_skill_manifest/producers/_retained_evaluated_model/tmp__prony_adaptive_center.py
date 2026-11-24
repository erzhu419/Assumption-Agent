import numpy as np
from scipy.optimize import nnls
import prony_grid_prior as bench
import prony_rheology as p


def estimate(t,s,y):
    tau=bench.TAU
    duration=np.ptp(t)
    d=np.c_[np.ones_like(t),(t-t.mean())/duration]
    a=bench.design(t,s,tau)
    base,r=p._calibrate(a,d,y)
    sig=p._noise(r)
    keep=np.abs(r-np.median(r))<=4.5*sig
    q=np.linalg.qr(d[keep],mode='reduced')[0]
    a=a[keep]-q@(q.T@a[keep]); b=y[keep]-q@(q.T@y[keep])
    mag=np.linalg.norm(a@base)/np.linalg.norm(s[keep]-s[keep].mean())
    fast=1/(1+(tau/(2*np.median(np.diff(t))))**2)
    out={'grid':base}
    for penalty in (12,):
        rows=(sig/mag)*np.c_[np.zeros(2),np.stack((penalty*tau/duration,fast))]
        aa=np.vstack((a,rows)); bb=np.r_[b,0.,0.]
        norms=np.linalg.norm(aa,axis=0)
        z=aa/norms
        g=nnls(z,bb,maxiter=1200)[0]/norms
        resid=bb-aa@g; best=resid@resid
        out[f'pen{penalty}']=g
        if penalty==12:
            out['both']=g
        def at(target):
            rhs=bb-aa[:,0]*target
            h=nnls(z[:,1:],rhs,maxiter=1200)[0]/norms[1:]
            e=rhs-aa[:,1:]@h
            return e@e,np.r_[target,h]
        for cutoff in (4.,9.):
            budget=best+cutoff*sig**2
            def endpoint(far):
                value,candidate=at(far)
                if value<=budget:
                    return candidate
                left,right=g[0],far
                candidate=g
                for _ in range(13):
                    v=(left+right)/2
                    loss,h=at(v)
                    if loss<=budget:
                        left,candidate=v,h
                    else:
                        right=v
                return candidate
            lo=endpoint(0.)
            upper=g[0]+mag
            while at(upper)[0]<=budget:
                upper=g[0]+2*(upper-g[0])
            hi=endpoint(upper)
            width = (hi[0] - lo[0]) / max(g[0], 1e-12 * mag)
            for threshold in (.10, .20, .35):
                fraction = .75 * width**2 / (width**2 + threshold**2)
                mid = (lo+hi)/2
                out[f'adaptive_{cutoff:g}_{threshold:g}'] = (1-fraction)*g+fraction*mid
                # The harmonic midpoint equalizes fractional under- and over-estimation.
                harmonic = 2*lo[0]*hi[0]/max(lo[0]+hi[0], 1e-12*mag)
                target = (1-fraction)*g[0]+fraction*harmonic
                out[f'relative_{cutoff:g}_{threshold:g}'] = at(target)[1]
    out['both_mass1.5']=out['adaptive_4_0.2']
    out['curved16_mass1.5']=out['relative_4_0.2']
    return out,keep

bench.estimate=estimate
bench.run()
