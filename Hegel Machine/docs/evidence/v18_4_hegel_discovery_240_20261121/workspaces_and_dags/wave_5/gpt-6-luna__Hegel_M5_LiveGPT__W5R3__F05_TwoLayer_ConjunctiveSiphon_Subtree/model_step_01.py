import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    a = np.asarray([[float(r['u_now']) / 4.0, float(r['s_grad'])] for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def pred(p, row):
        u = float(row['u_now']) / 4.0
        s = float(row['s_grad'])
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(max(u, 0.0))
    def residual(p):
        return np.asarray([pred(p, r) for r in train_rows]) - y
    p0 = np.array([6.5, 0.70, 0.15])
    res = least_squares(residual, p0, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    vmax, rho, c = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(max(u, 0.0)))

_TRAIN_ROWS = []
_original_fit = fit
def fit(train_rows):
    global _TRAIN_ROWS
    _TRAIN_ROWS = list(train_rows)
    return _original_fit(train_rows)
