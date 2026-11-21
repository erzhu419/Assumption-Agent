import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def fit(train_rows):
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        u, s = _features(r)
        vmax, rho, c_loc = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base + c_loc * math.sqrt(u)
    def residual(p):
        return np.asarray([pred(p, r) for r in rows]) - ys
    best = None
    for start in ([6.5, 0.7, 0.0], [8.0, 0.4, -0.2], [5.0, 0.8, 0.2]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    vmax, rho, c_loc = fit(train_rows)
    u, s = _features(row)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))