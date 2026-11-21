import math
import numpy as np
from scipy.optimize import least_squares

def _value(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    lo = np.array([0.0, 0.0, 0.0, 0.0, 2.50])
    hi = np.array([20.0, 2.0, 10.0, 10.0, 2.68])
    starts = [np.array([4.0, 0.5, 1.0, 1.0, sc]) for sc in (2.51, 2.59, 2.67)]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(p, row))