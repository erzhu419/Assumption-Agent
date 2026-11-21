import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(0.0, float(row['u_now'])) / 4.0
    s = max(0.0, float(row['s_grad']))
    base = (1.0 - s / (1.0 + s)) * (u / (1.2 + u))
    return u, s, base

def _pred(row, p):
    u, s, base = _features(row)
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_pred(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.2]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    return float(_pred(row, fit([row])))