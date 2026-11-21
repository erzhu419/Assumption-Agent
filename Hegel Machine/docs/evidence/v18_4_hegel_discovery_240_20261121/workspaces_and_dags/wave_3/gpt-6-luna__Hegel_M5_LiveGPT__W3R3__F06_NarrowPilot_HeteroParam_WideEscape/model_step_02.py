import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = (u / (1.2 + u))
    return u, s, base

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        vmax, rho, c = p
        u, s, base = _features(r)
        return vmax * (1.0 - rho * s / (1.0 + s)) * base + c * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.2]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u, s, base = _features(row)
    vmax, rho, c = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * base + c * math.sqrt(u))