import json
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _features, _invert, _jacobian

s = pickle.load(open('public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
n = len(x)
i, j = np.tril_indices(4, -1)
axis = np.count_nonzero(x, axis=1) == 1
order = np.argsort(np.argsort(np.linalg.norm(x, axis=1)))
splits = dict(interleaved=(np.arange(n) % 5, range(5)), amplitude=(order * 5 // n, range(5)), axis=(axis, [True]), sweep_order=(np.arange(n) * 5 // n, range(5)))

def design(x, mode):
    b = _features(x)
    pair_total = x[:, i]**2 + x[:, j]**2
    if mode == 'additive':
        return b[:, :9]
    if mode == 'rational-extra':
        return np.c_[b, x**5 / (1 + x*x), b[:, 9:] * pair_total / (1 + pair_total)]
    if mode == 'degree3-extra':
        return np.c_[b, x*x, *[x[:, a]**2*x[:, d] for a in range(4) for d in range(4) if a != d], *[np.prod(np.delete(x, a, axis=1), axis=1) for a in range(4)]]
    return b

def fit(z, yy):
    w = np.ones(len(yy))
    for _ in range(20):
        b = np.linalg.lstsq(z*w[:, None], yy*w[:, None], rcond=None)[0]
        e = np.linalg.norm(yy-z@b, axis=1)
        w = 1/(1+(e/max(2*np.median(e), 1e-12))**2)
    return b

# Label gross artifacts using baseline out-of-fold residuals, never in the fit.
cv = np.empty(n)
for k in range(5):
    te = np.arange(n) % 5 == k
    cv[te] = np.linalg.norm(_features(x[te]) @ _fit(x[~te], y[~te]) - y[te], axis=1)
gross = cv > 5
print('gross EMI rows:', np.flatnonzero(gross).tolist(), 'L2:', cv[gross], 'clean max:', cv[~gross].max())
print('sweep metadata keys:', list(c), 'study batch/relay keys:', [k for k in s if 'batch' in k or 'relay' in k])
log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
print('telemetry structure:', list(log) if isinstance(log, dict) else (type(log).__name__, len(log)))
print('difference basis identity max error:', np.max(np.abs(((x[:,i]+x[:,j])**2-(x[:,i]-x[:,j])**2)/4-x[:,i]*x[:,j])))
for name, (groups, folds) in splits.items():
    nums = {}
    for mode in ('additive', 'pair', 'rational-extra', 'degree3-extra'):
        z = design(x, mode)
        errors = []
        for k in folds:
            te = groups == k
            test = te & ~gross
            b = fit(z[~te], y[~te])
            errors.extend(np.linalg.norm(z[test]@b-y[test],axis=1))
        nums[mode] = tuple(float(f'{q:.4g}') for q in (np.median(errors), max(errors)))
    inv, bias = [], []
    for k in folds:
        te = groups == k
        test = te & ~gross
        d = _invert(y[test], _fit(x[~te], y[~te])) - x[test]
        inv.extend(np.linalg.norm(d, axis=1))
        bias.append(np.max(np.abs(d.mean(axis=0))))
    print(name, 'forward median/max', nums, 'pair inverse max', max(inv), 'max fold axis bias', max(bias))
coef = _fit(x, y)
obs = _invert(s['y_obs_raw'], coef).reshape(-1, 4)
sv = np.linalg.svd(_jacobian(np.vstack((obs, x[~gross])), coef), compute_uv=False)
print('observed ref range:', obs.min(axis=0), obs.max(axis=0), 'clean sweep ref range:', x[~gross].min(axis=0), x[~gross].max(axis=0))
print('minimum Jacobian singular value:', sv[:, -1].min(), 'maximum condition:', (sv[:, 0]/sv[:, -1]).max())
