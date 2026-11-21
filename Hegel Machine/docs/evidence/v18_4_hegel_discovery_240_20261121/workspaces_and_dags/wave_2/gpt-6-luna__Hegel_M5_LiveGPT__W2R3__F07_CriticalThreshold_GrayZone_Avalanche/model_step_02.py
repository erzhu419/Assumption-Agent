import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(u, 0.0))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    lo = np.array([1.0, 0.05, -2.0, 0.0, 2.36])
    hi = np.array([15.0, 0.95, 2.0, 10.0, 4.5])
    starts = [
        [6.5, 0.70, 0.15, 2.0, 2.6],
        [5.0, 0.65, 0.5, 1.0, 2.5],
        [8.0, 0.75, -0.2, 3.0, 2.8],
        [6.0, 0.70, 0.0, 5.0, 3.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, np.clip(start, lo + 1e-8, hi - 1e-8),
                               bounds=(lo, hi), max_nfev=5000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return float(_value(row, _PARAMS))

# The fitted parameters are retained for subsequent predictions.
_PARAMS = None
_fit = fit
def fit(train_rows):
    global _PARAMS
    _PARAMS = _fit(train_rows)
    return _PARAMS
