import contextlib, io, sys
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/chamber_probe.py').read())
import downstream_solver as m
from scipy.optimize import minimize_scalar
from collections import defaultdict

# Use the same reference moments and imputation as prediction on deployment coupons.
X, _ = m._features(tr, te)
pasts, holds, feats = m._inputs(tr, X)
labels = np.asarray(s['lab_id'])
truth = np.c_[Y, R]
results = defaultdict(list)
for lab in np.unique(labels):
    keep = np.asarray(tr['lab_id']) != lab
    tau_l, th_l = m._fit(pasts[keep], holds[keep], feats[keep], dt, t, (tr['strain_curves_obs']-tr['strain_0'][:,None])[keep], tr['instant_rate_obs'][keep])
    pp, hh, ff = m._inputs(s, x)
    vc, vr = m._design(tau_l, pp, hh, ff, dt, t)
    branches = np.concatenate((vc[:,:,:2], vr[:,None,:2]), axis=1)
    base = branches @ th_l[:2]
    drift = ff @ th_l[2:]
    val = labels == lab
    def pred(kind, h):
        if kind in ('unit', 'exponential', 'memory difference'):
            g = -h*np.expm1(-t/h)
        elif kind == 'rational':
            g = t/(1+t/h)
        elif kind == 'quadratic denominator':
            g = t/(1+(t/h)**2)
        else:
            g = t
        return base + drift[:,None]*np.r_[g,1.]
    for kind in ('unit', 'exponential', 'rational', 'quadratic denominator', 'memory difference', 'linear'):
        h = 1. if kind in ('unit','linear') else minimize_scalar(lambda q: np.mean((pred(kind,q)[~val]-truth[~val])**2),bounds=(.2,8.),method='bounded').x
        p = pred(kind,h)
        if kind == 'memory difference':
            diff = branches[:,:,0]*th_l[0]-branches[:,:,1]*th_l[1]
            c = np.sum(diff[~val]*(truth-p)[~val])/np.sum(diff[~val]**2)
            p = p+c*diff
        e = np.abs((truth-p)[val])
        results[kind].append((np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9),np.quantile(e[:,-2],.9),e[:,:-1].max()))
    print('lab',lab,'diagnostic rows',val.sum(),'unit terminal bias',round(float(np.median((truth-pred('unit',1.))[val,-2])),5))
for kind, vals in results.items():
    print(kind, 'worst-lab strain P90 / rate P90 / terminal P90 / strain max',np.round(np.max(vals,axis=0),6))
