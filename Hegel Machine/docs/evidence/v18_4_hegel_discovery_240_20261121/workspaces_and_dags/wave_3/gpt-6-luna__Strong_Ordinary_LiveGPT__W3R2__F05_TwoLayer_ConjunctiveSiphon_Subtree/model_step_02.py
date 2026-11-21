import math
import numpy as np
from scipy.optimize import least_squares

# Fit the documented two-layer siphon base with each plausible open-branch
# form, then retain the numerically best-fitting branch.

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # supplied tau_scale = 1.0
    return un, s

def _value(row, p, branch):
    un, s = _features(row)
    vmax, rho = p[0], p[1]
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt':
        return base + p[2] * math.sqrt(max(0.0, un))
    return base - p[2] * un

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    candidates = []
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in train_rows]) - ys
        result = least_squares(residual, x0=[5.0, 0.5, 0.0],
                               bounds=([0.0, 0.0, -20.0], [100.0, 1.0, 20.0]),
                               max_nfev=20000)
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, branch, params = min(candidates, key=lambda item: item[0])
    return {'branch': branch, 'params': np.asarray(params, dtype=float)}

def predict(row):
    if not hasattr(predict, '_fit'):
        raise RuntimeError('predict requires fitted parameters')
    model = predict._fit
    return float(_value(row, model['params'], model['branch']))

# fit() stores the selected parameters for the required predict(row) interface.
_fit_impl = fit
def fit(train_rows):
    model = _fit_impl(train_rows)
    predict._fit = model
    return model