import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    # Saturating local response plus a gradient-dependent two-hop transfer term.
    def pred(p, r):
        u, s = _features(r)
        a, b, c, d = p
        return a*u/(1.0+b*u) + c*u*(1.0-s) + d*u*u*(1.0-s)
    def resid(p):
        return np.array([pred(p, r) for r in rows]) - y
    best = None
    for start in ([2.0, 0.3, 0.2, 0.0], [3.0, 0.5, 0.5, -0.05], [1.0, 0.1, 0.1, 0.02]):
        res = least_squares(resid, np.array(start), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    u, s = _features(row)
    a, b, c, d = _PARAMS
    return float(a*u/(1.0+b*u) + c*u*(1.0-s) + d*u*u*(1.0-s))