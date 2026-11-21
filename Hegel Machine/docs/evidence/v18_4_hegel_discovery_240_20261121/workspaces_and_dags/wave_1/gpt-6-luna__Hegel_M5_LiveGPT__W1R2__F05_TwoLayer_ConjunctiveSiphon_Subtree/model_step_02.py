import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 4.0

def _predict_params(row, params):
    vmax, rho, c_loc = params
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_predict_params(row, _PARAMS))