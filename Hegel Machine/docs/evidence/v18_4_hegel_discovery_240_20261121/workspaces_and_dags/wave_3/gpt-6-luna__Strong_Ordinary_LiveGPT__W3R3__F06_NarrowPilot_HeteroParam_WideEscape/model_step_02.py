import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return un, s

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        vmax, rho, c_loc = p
        u, s = _features(r)
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(train_rows)
    u, s = _features(row)
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))