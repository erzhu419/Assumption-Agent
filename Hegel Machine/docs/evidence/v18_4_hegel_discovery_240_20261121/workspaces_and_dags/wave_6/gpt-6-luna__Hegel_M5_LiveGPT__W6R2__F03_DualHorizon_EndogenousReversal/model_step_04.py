import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    h = float(row['H'])
    return u / (1.2 + u), u, h

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, row):
        f, u, h = _features(row)
        vmax, amp, tf, ts = p
        return vmax * f * (-math.expm1(-h / tf)) - amp * u * (-math.expm1(-h / ts))
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    starts = [[1.0, 1.0, 0.3, 20.0], [2.0, 0.2, 0.5, 10.0], [1.0, 0.1, 0.2, 30.0]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0, 0, 1e-5, 1e-4], [100, 100, 100, 1e4]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = float(row['u_now']); h = float(row['H'])
    return float(p[0] * (u / (1.2 + u)) * (-math.expm1(-h / p[2])) - p[1] * u * (-math.expm1(-h / p[3])))