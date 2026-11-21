import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 4.0

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_predict_params(row, fit(train_rows)))