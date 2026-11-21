import math
import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    branch = c_loc * math.sqrt(max(u, 0.0))
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + branch - avalanche

def fit(train_rows):
    global _fit_params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [6.5, 0.70, 0.10, 1.0, 1.3],
        [6.5, 0.70, -0.10, 1.0, 1.5],
        [8.0, 0.70, 0.0, 1.5, 1.0],
    ]
    lo = [0.01, 0.05, -5.0, 0.0, 0.0]
    hi = [30.0, 0.95, 5.0, 10.0, 3.0]
    results = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    best = min(results, key=lambda result: float(np.dot(result.fun, result.fun)))
    _fit_params = best.x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _fit_params))