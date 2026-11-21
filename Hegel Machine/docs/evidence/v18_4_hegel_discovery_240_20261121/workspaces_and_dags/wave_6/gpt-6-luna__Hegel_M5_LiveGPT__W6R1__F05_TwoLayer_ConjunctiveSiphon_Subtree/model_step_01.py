import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(p, r) for r in rows]) - y
    p0 = np.array([6.5, 0.70, 0.15], dtype=float)
    result = least_squares(residual, p0, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    return float(_predict_params(fit(_TRAIN_ROWS), row))