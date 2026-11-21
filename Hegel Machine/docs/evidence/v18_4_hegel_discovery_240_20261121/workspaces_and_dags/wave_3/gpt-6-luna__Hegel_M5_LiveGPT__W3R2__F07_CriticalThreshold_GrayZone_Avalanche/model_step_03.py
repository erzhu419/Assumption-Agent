import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    starts = [
        [6.5, 0.7, 0.1, 1.0, 2.58],
        [8.0, 0.68, 0.5, 2.0, 2.62],
        [5.0, 0.75, -0.3, 0.5, 2.52],
        [10.0, 0.66, 1.0, 3.0, 2.67],
    ]
    lower = [1.0, 0.05, -2.0, 0.0, 2.50]
    upper = [15.0, 0.95, 2.0, 10.0, 2.68]
    best = None
    for p0 in starts:
        result = least_squares(
            lambda p: np.array([_value(p, r) for r in rows]) - y,
            p0, bounds=(lower, upper), max_nfev=20000
        )
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS.tolist()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))