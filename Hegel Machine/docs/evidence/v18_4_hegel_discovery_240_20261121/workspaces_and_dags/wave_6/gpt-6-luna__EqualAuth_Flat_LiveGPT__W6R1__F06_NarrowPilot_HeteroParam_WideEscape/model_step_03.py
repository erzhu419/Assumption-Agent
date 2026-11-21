import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(p, row):
    vmax, rho, km, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u))
    local = c_loc * math.sqrt(u)
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + local - avalanche

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_pred(p, r) for r in train_rows]) - y
    starts = [
        [5.0, 0.7, 1.0, 0.4, 0.2, 2.6],
        [7.0, 0.9, 1.5, 0.2, 0.5, 2.5],
        [4.0, 0.5, 0.5, 0.8, 0.1, 2.7],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0,
            bounds=([0.01, 0.0, 0.01, -5.0, 0.0, 0.0],
                    [100.0, 1.5, 100.0, 5.0, 20.0, 10.0]),
            max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return {'params': [float(v) for v in _PARAMS]}

def predict(row):
    p = row.get('_params') if isinstance(row, dict) else None
    if p is None:
        p = _PARAMS
    if p is None:
        raise ValueError('fit must be called before predict')
    if isinstance(p, dict):
        p = p.get('params')
    return float(_pred(p, row))