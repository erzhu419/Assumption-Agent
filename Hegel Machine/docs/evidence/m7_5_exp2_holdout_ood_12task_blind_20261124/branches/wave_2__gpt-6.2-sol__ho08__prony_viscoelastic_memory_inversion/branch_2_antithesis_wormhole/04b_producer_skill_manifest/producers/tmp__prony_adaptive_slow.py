import numpy as np
from scipy.optimize import nnls
import prony_grid_prior as bench
import prony_rheology as p


def estimate(t, s, y):
    tau = bench.TAU
    duration = np.ptp(t)
    drift = np.c_[np.ones_like(t), (t-t.mean())/duration]
    a = bench.design(t,s,tau)
    g, r = p._calibrate(a,drift,y)
    sig = p._noise(r)
    keep = abs(r-np.median(r)) <= 4.5*sig
    q = np.linalg.qr(drift[keep],mode='reduced')[0]
    b = y[keep] - q@(q.T@y[keep])
    ap = a[keep] - q@(q.T@a[keep])
    mag = np.linalg.norm(ap@g)/np.linalg.norm(s[keep]-s[keep].mean())
    fast = 1/(1+(tau/(2*np.median(np.diff(t))))**2)

    def solve(strength, cutoff=np.inf):
        rows = (sig/mag)*np.c_[np.zeros(2),np.stack((strength*tau/duration,fast))]
        aa = np.vstack((ap,rows)); bb = np.r_[b,0.,0.]
        m = np.r_[True, tau<=cutoff]
        aa = aa[:,m]
        scale = np.linalg.norm(aa,axis=0)
        coefficients = nnls(aa/scale,bb,maxiter=1200)[0]/scale
        full = np.zeros(len(m)); full[m] = coefficients
        residual = bb-aa@coefficients
        return full,residual@residual

    v02,_ = solve(12)
    _,best = solve(0)
    out = {'grid':g, 'v02':v02}
    for frac in (.30,.40,.50):
        _,short = solve(0,max(tau[0],frac*duration))
        evidence = max(0.,(short-best)/sig**2)
        for threshold in (4.,9.,16.):
            strength = 4. + 8./(1.+evidence/threshold)
            name = f'evidence{frac:g}_{threshold:g}'
            out[name] = solve(strength)[0]
    out['both'] = v02
    out['both_mass1.5'] = out['evidence0.4_4']
    out['curved16_mass1.5'] = out['evidence0.4_9']
    return out,keep

bench.estimate=estimate
bench.run()
