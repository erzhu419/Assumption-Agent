import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    lo = np.array([1.0, 0.05, -2.0, 0.0, 2.36])
    hi = np.array([15.0, 0.95, 2.0, 5.0, 3.0])
    starts = [
        [6.5, 0.70, 0.0, 1.0, 2.55],
        [8.0, 0.72, -0.3, 1.5, 2.65],
        [5.0, 0.65, 0.5, 0.8, 2.45],
        [9.0, 0.75, -0.8, 2.0, 2.75],
    ]
    best = None
    for x0 in starts:
        result = least_squares(lambda p: np.asarray([_value(r, p) for r in rows]) - y,
                               np.clip(x0, lo + 1e-8, hi - 1e-8), bounds=(lo, hi),
                               max_nfev=30000, x_scale='jac')
        cost = float(np.sum(result.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))