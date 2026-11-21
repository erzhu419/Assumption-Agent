import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _features(row):
    u = max(float(row['u_now']), 1e-12)
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    grad = s / (1.0 + s)
    return base, base * grad, u, math.sqrt(u)

def fit(train_rows):
    global _PARAMS, _BRANCH
    branches = ('plus_sqrt', 'minus_sqrt', 'leak')
    best = None
    for branch in branches:
        def pred_params(p, row):
            base, bg, u, rootu = _features(row)
            v, vrho, c = p
            term = c * rootu if branch == 'plus_sqrt' else (-c * rootu if branch == 'minus_sqrt' else -c * u)
            return v * base - vrho * bg + term
        def residual(p):
            return np.array([pred_params(p, r) - float(r['y']) for r in train_rows])
        result = least_squares(residual, x0=np.array([2.0, 0.5, 0.1]), max_nfev=5000)
        rss = float(np.dot(result.fun, result.fun))
        if best is None or rss < best[0]:
            best = (rss, branch, result.x.copy())
    _, _BRANCH, _PARAMS = best
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called first')
    base, bg, u, rootu = _features(row)
    v, vrho, c = _PARAMS
    if _BRANCH == 'plus_sqrt':
        term = c * rootu
    elif _BRANCH == 'minus_sqrt':
        term = -c * rootu
    else:
        term = -c * u
    return float(v * base - vrho * bg + term)
