import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_KIND = None

def _features(row):
    u = max(float(row['u_now']) / 0.05, 1e-8)
    s = max(float(row['s_grad']), 0.0)
    return u, s

def _predict_kind(row, kind, p):
    u, s = _features(row)
    if kind == 'hill':
        vmax, k_half, alpha, leak = p
        return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha*s) - leak*u
    j0, beta, rho, c_sqrt = p
    return j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c_sqrt*math.sqrt(u)

def fit(train_rows):
    global _PARAMS, _KIND
    ys = np.asarray([float(r['y']) for r in train_rows])
    best = None
    # Numerically fit both four-parameter stationary mechanisms documented for this subtree.
    starts = {
        'hill': ([5.0, 2.0, 0.5, 0.05], [3.0, 10.0, 0.2, 0.1]),
        'barrier': ([8.0, 1.0, 0.7, 0.2], [15.0, 3.0, 0.5, 0.1])
    }
    bounds = {
        'hill': ([0.0, 1e-8, 0.0, 0.0], [100.0, 1e4, 100.0, 10.0]),
        'barrier': ([0.0, 0.0, 0.0, -10.0], [100.0, 100.0, 0.999, 10.0])
    }
    for kind in ('hill', 'barrier'):
        lo, hi = bounds[kind]
        for x0 in starts[kind]:
            result = least_squares(lambda p: np.asarray([_predict_kind(r, kind, p) for r in train_rows])-ys,
                                   x0=np.asarray(x0), bounds=(lo, hi), max_nfev=30000)
            err = float(np.sum(result.fun**2))
            if best is None or err < best[0]:
                best = (err, kind, result.x.copy())
    _PARAMS, _KIND = best[2], best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_kind(row, _KIND, _PARAMS))