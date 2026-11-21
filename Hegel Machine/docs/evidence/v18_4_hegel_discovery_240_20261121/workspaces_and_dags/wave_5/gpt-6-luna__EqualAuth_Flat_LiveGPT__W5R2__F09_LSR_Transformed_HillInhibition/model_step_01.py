import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 2.5

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    s_grad = float(row['s_grad'])
    hill = (u_n ** 2) / (k_half + u_n ** 2)
    return vmax * hill / (1.0 + alpha_inh * s_grad) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    def residuals(p):
        return np.asarray([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residuals, x0=np.array([5.0, 2.0, 0.2, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [100.0, 1000.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))