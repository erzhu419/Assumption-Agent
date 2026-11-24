import numpy as np
from scipy.optimize import nnls
import prony_grid_prior as bench
import prony_rheology as p


def estimate(t, s, y):
    tau = bench.TAU
    duration = np.ptp(t)
    drift = np.c_[np.ones_like(t), (t-t.mean())/duration]
    a = bench.design(t, s, tau)
    g, r = p._calibrate(a, drift, y)
    noise = p._noise(r)
    keep = abs(r-np.median(r)) <= 4.5*noise
    q = np.linalg.qr(drift[keep], mode='reduced')[0]
    b = y[keep] - q@(q.T@y[keep]); ap = a[keep] - q@(q.T@a[keep])
    mag = np.linalg.norm(ap@g)/np.linalg.norm(s[keep]-s[keep].mean())
    fast = 1/(1+(tau/(2*np.median(np.diff(t))))**2)
    x = tau/duration
    # Match the linear penalty at one third of the observation duration,
    # while testing whether its growth beyond that point hurts genuine slow modes.
    profiles = {
        'v02': 12*x,
        'linear10': 10*x,
        'sqrt': 4*np.sqrt(3*x),
        'sat_soft': 4*(1+1.5/3)*3*x/(1+1.5*x),
        'sat_mid': 4*(1+4/3)*3*x/(1+4*x),
        'sat_hard': 4*(1+9/3)*3*x/(1+9*x),
        'cap': 4*np.minimum(3*x,1.),
    }
    result = {'grid': g}
    for name, slow in profiles.items():
        rows = (noise/mag)*np.column_stack((np.zeros(2), (slow, fast)))
        aa = np.vstack((ap, rows)); norm = np.linalg.norm(aa, axis=0)
        result[name] = nnls(aa/norm, np.r_[b,0.,0.], maxiter=1200)[0]/norm
    for name, key in [('both','v02'),('both_mass1.5','sat_mid'),('curved16_mass1.5','cap')]:
        result[name] = result[key]
    return result, keep

bench.estimate = estimate
bench.run()
