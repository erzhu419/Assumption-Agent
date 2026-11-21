import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.05

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.5]),
                           bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_predict_params(row, p))