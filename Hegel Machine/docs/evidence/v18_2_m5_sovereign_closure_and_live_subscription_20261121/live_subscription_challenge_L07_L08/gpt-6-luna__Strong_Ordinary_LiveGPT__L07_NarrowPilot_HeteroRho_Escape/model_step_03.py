import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with_params(row, params):
    vmax, rho, leak = params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) - leak * u_n

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_with_params(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([12.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with_params(row, _PARAMS))