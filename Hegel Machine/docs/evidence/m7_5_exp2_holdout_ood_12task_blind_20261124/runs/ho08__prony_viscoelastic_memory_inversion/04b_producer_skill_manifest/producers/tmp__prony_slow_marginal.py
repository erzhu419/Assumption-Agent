import numpy as np
from scipy.optimize import nnls
from scipy.special import softmax
import prony_grid_prior as bench
import prony_rheology as p


def estimate(t, s, y):
    duration = np.ptp(t)
    tau = bench.TAU
    drift = np.c_[np.ones_like(t), (t-t.mean())/duration]
    a = bench.design(t, s, tau)
    g, r = p._calibrate(a, drift, y)
    noise = p._noise(r)
    keep = abs(r-np.median(r)) <= 4.5*noise
    q = np.linalg.qr(drift[keep], mode='reduced')[0]
    b = y[keep] - q@(q.T@y[keep]); ap = a[keep] - q@(q.T@a[keep])
    mag = np.linalg.norm(ap@g)/np.linalg.norm(s[keep]-s[keep].mean())
    fast = 1/(1+(tau/(2*np.median(np.diff(t))))**2)
    rhs = np.r_[b, 0., 0.]
    def solve(use, strength):
        rows = (noise/mag)*np.c_[np.zeros(2), np.stack((strength*tau/duration,fast))]
        aa = np.vstack((ap, rows))[:,use]
        norms = np.linalg.norm(aa, axis=0)
        coeff = nnls(aa/norms,rhs,maxiter=1200)[0]/norms
        residual = rhs-aa@coeff
        full = np.zeros(len(tau)+1); full[use] = coeff
        return full, residual@residual/noise**2
    v02,_ = solve(np.arange(len(tau)+1),12)
    out = {'grid':g, 'v02':v02, 'both':v02}
    for split in (duration/12, duration/8):
        short = np.r_[0,1+np.flatnonzero(tau<=split)]
        long = 1+np.flatnonzero(tau>split)
        if not len(long):
            continue
        for strength in (0,6):
            fits = [solve(np.r_[short,j],strength) for j in long]
            coefs, scores = map(np.array, zip(*fits))
            for temp in (1., 3.):
                weights = softmax(-.5*(scores-min(scores))/temp)
                mid = weights@coefs
                for fraction in (.5,1.):
                    out[f'marginal_{split/duration:.3f}_{strength}_{temp:g}_{fraction:g}'] = (1-fraction)*v02+fraction*mid
    out['both_mass1.5']=out['marginal_0.125_6_1_0.5']
    out['curved16_mass1.5']=out['marginal_0.125_0_1_0.5']
    return out,keep

bench.estimate=estimate
bench.run()
