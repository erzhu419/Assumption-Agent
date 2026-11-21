import math
import numpy as np
from scipy.optimize import least_squares

# Fit both archived static siphon candidates, including either sign of the
# square-root branch, and retain the lowest-residual numerical fit.
_PARAMS = None
_KIND = None

def _value(kind, p, row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    if kind == 'C':
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(u)
    j0, beta, rho, c = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _PARAMS, _KIND
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for kind, n in [('C', 3), ('R', 4)]:
        for sign in (-1.0, 1.0):
            def resid(p):
                q = p.copy()
                q[-1] *= sign
                return np.array([_value(kind, q, r) for r in train_rows]) - y
            x0 = np.array([4.0, 0.7, 0.2] if kind == 'C' else [5.0, 0.5, 0.7, 0.2])
            lo = np.array([0.0, 0.0, 0.0] if kind == 'C' else [0.0, 0.0, 0.0, 0.0])
            hi = np.array([100.0, 4.0, 10.0] if kind == 'C' else [100.0, 20.0, 4.0, 10.0])
            res = least_squares(resid, x0, bounds=(lo, hi), max_nfev=20000)
            err = float(np.sum(res.fun ** 2))
            if best is None or err < best[0]:
                q = res.x.copy()
                q[-1] *= sign
                best = (err, kind, q)
    _, _KIND, _PARAMS = best
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_value(_KIND, _PARAMS, row))