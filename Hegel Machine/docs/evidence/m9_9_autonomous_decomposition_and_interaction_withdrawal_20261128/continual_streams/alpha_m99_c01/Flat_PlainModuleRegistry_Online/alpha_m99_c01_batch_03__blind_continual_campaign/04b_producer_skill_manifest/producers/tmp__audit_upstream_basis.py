import pickle
import sys
import numpy as np
from scipy.optimize import root
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _invert

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
# Gross spikes are excluded from scoring, not removed from the candidate's training folds.
clean = (np.abs(y).max(axis=1) < 10)
axis_only = (np.count_nonzero(x, axis=1) == 1)
large = (np.abs(x).max(axis=1) >= 1.4)

# Shifting polynomial variables to pair sums/differences adds no cubic basis terms.
# The fixed rational challengers test saturation outside that polynomial span.
def feature(z, a, kind):
    q = z if kind == 'axis' else (z - np.roll(z, -1, axis=-1)) / np.sqrt(2.)
    return q**3 / (1. + a*q*q)

variants = [('axis', .25), ('axis', 1.), ('pair difference', .25), ('pair difference', 1.)]
fns = [lambda z, k=k, a=a: feature(z, a, k) for k, a in variants]
pred = np.empty((1+len(fns), len(x), 4)) * np.nan
for i in np.flatnonzero(clean):
    keep = np.arange(len(x)) != i
    c = _fit_bridge(x[keep], y[keep])
    pred[0, i] = _invert(y[i], c)
    # Classify spikes by the training fit only. Keep the candidate's supported
    # polynomial columns and refit them jointly with the rational additions.
    err = np.linalg.norm(_basis(x[keep]) @ c - y[keep], axis=1)
    good = keep.copy()
    good[keep] = err <= max(8*np.median(err), 1e-9)
    for v, phi in enumerate(fns, 1):
        design = np.c_[_basis(x[good]), phi(x[good])]
        b = np.zeros((design.shape[1], 4))
        for j in range(4):
            use = np.r_[np.arange(35) < 5, np.ones(4, dtype=bool)]
            use[:35] |= c[:, j] != 0
            b[use, j] = np.linalg.lstsq(design[:, use], y[good, j], rcond=None)[0]
        def resid(z):
            return np.r_[_basis(z), phi(z)] @ b - y[i]
        fit = root(resid, pred[0, i], options={'xtol': 1e-10})
        if np.max(np.abs(fit.fun)) > 1e-8:
            raise AssertionError(('challenger inversion', v, i))
        pred[v, i] = fit.x
for v, name in enumerate(['sparse cubic'] + [str(t) for t in variants]):
    e = pred[v]-x
    print(name, 'worst inverse error by slice (axis, mixed, extreme):',
          [float(np.max(np.abs(e[clean & a]))) for a in (axis_only, ~axis_only, large)],
          'RMSE', np.sqrt(np.mean(e[clean]**2)),
          'worst total-load error', np.max(np.abs(e[clean].sum(axis=1))))
print('Dataset keys:', sorted(ds))
print('Calibration fields:', sorted(cal))
print('Train/test lab groups:', [dict(zip(*np.unique(ds[k]['lab_id'], return_counts=True))) for k in ('train','test')])
assert np.max(np.abs(pred[0, clean]-x[clean])) < .001
