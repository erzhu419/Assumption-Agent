import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale = 1.0
    base = un / (1.2 + un)
    return un, s, base

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    un, s, base = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * base + c_loc * math.sqrt(max(un, 0.0)) - k_aval * max(0.0, s - s_crit) * un

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    best = None
    starts = [
        [5.0, 0.7, 0.1, 1.0, 2.6],
        [8.0, 0.5, 0.5, 1.5, 2.55],
        [4.0, 0.8, -0.2, 0.8, 2.65],
        [10.0, 0.6, 0.0, 2.0, 2.5],
    ]
    lo = [0.0, 0.0, -10.0, 0.0, 2.0]
    hi = [100.0, 1.5, 10.0, 20.0, 3.2]
    for p0 in starts:
        result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return float(_value(row, fit([dict(row, y=0.0)])))