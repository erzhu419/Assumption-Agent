import numpy as np
from scipy.optimize import nnls
import prony_grid_prior as bench
import prony_rheology as p

# All penalties are in units of the measured noise. An ultra-fast mode
# contributes very little to a training ramp, even at substantial modulus.
spec = [(0, 0, 0), (6, 0, 0), (12, 0, 0), (10, .5, 0)]
spec += [(k, 0, f) for k in (6, 9, 12) for f in (.5, 1., 2.)]
spec += [(k, .5, 1.) for k in (6, 9, 12)]

def estimate(t, s, y):
    tau = bench.TAU
    drift = np.c_[np.ones_like(t), (t-t.mean())/np.ptp(t)]
    a = bench.design(t, s, tau)
    g, r = p._calibrate(a, drift, y)
    noise = p._noise(r)
    keep = abs(r-np.median(r)) <= 4.5*noise
    q = np.linalg.qr(drift[keep], mode='reduced')[0]
    b = y[keep] - q @ (q.T @ y[keep])
    ap = a[keep] - q @ (q.T @ a[keep])
    mag = max(np.linalg.norm(ap @ g)/np.linalg.norm(s[keep]-s[keep].mean()), np.finfo(float).eps)
    slow = tau/np.ptp(t)
    fast = 1/(1+(tau/(2*np.median(np.diff(t))))**2)
    result = {'grid': g}
    for k, mass, f in spec[1:]:
        rows = (noise/mag)*np.column_stack((np.zeros(3), np.stack((k*slow, np.full_like(tau,mass), f*fast))))
        aa = np.vstack((ap, rows))
        norm = np.maximum(np.linalg.norm(aa, axis=0), 1e-14)
        result[f's{k}_m{mass:g}_f{f:g}'] = nnls(aa/norm, np.r_[b,0.,0.,0.], maxiter=1200)[0]/norm
    # Preserve these names for the existing fast-present/absent report.
    for name, key in [('both','s6_m0_f0'),('both_mass1.5','s9_m0_f1'),('curved16_mass1.5','s9_m0.5_f1')]:
        result[name] = result[key]
    return result, keep

bench.estimate = estimate
bench.run()
