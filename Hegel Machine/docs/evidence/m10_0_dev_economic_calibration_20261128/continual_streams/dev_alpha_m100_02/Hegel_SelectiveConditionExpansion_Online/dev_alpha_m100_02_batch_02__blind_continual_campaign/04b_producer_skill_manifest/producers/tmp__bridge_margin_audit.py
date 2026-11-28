import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
idx = np.arange(len(x))
p = u._fit_bridge(x, y)
spikes = np.linalg.norm(u._forward(x, p) - y, axis=1) > 1
folds = list(np.split(idx[:16], 4)) + list(np.array_split(idx[16:], 8))

def response(z, q, kind, sign):
    base = u._forward(z, q[:12])
    if kind == 'base':
        return base
    contrast = z + sign * z[:, u._NEXT]
    if kind == 'contrast cubic':
        return base + q[4:8] * q[12] * contrast**3
    return q[:4] + (base - q[:4]) / (1 + q[12] * contrast**2)

models = [('base', 0), ('contrast cubic', -1), ('contrast cubic', 1), ('rational', -1), ('rational', 1)]
errors = {model: np.zeros_like(x) for model in models}
forward = {model: np.zeros(len(x)) for model in models}
for g in folds:
    train = ~np.isin(idx, g)
    q0 = u._fit_bridge(x[train], y[train])
    k = g[~spikes[g]]
    a, b = x[train], y[train]
    residual = np.linalg.norm(u._forward(a, q0) - b, axis=1)
    clean = residual < max(8 * np.median(residual), 0.002)
    a, b = a[clean], b[clean]
    for model in models:
        kind, sign = model
        q = q0
        if kind != 'base':
            lower = np.r_[np.full(12, -np.inf), -0.04 if kind == 'rational' else -np.inf]
            q = least_squares(lambda w: (response(a, w, kind, sign) - b).ravel(), np.r_[q0, 0.], bounds=(lower, np.inf), x_scale='jac').x
        forward[model][k] = np.linalg.norm(response(x[k], q, kind, sign) - y[k], axis=1)
        initial = u._inverse(y[k], q0)
        if kind == 'base':
            recovered = initial
        else:
            recovered = np.array([least_squares(lambda z: response(z[None], q, kind, sign)[0] - target, start, gtol=1e-13).x for start, target in zip(initial, y[k])]).reshape(-1, 4)
        errors[model][k] = recovered - x[k]
for model in models:
    e = errors[model]
    norms = np.linalg.norm(e, axis=1)
    print(model, 'worst axis/mixed/total/forward', norms[:16].max(), norms[16:].max(), np.abs(e.sum(axis=1)).max(), forward[model].max())

# Bound row diagonal dominance of the gain-normalized Jacobian on the observed load box.
out = u.calibrate_upstream(d)
observed = [x.ravel()]
for key in ('train', 'test', 'chamber_verification_trials'):
    part = out.get(key)
    if part is not None:
        observed.extend(part[f].ravel() for f in ('past_channel_raw', 'hold_channel_raw'))
lo = min(z.min() for z in observed)
hi = max(z.max() for z in observed)
s, t, m, v = p[8:]
# |s + m*z| is a maximum of two affine forms. Minimize each resulting quadratic exactly.
assert v > 0
margin = min(1 + min(m*lo, m*hi) + 3*v*z*z - abs(t) - e*(s + m*z)
             for e in (-1, 1) for z in (np.clip(e*m/(6*v), lo, hi),))
print('load box', (lo, hi), 'Jacobian dominance margin', margin, 'bridge fields', sorted(c))
assert margin > 0
