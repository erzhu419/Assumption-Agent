import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.1

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    hill = (u_n ** 2) / (k_half + u_n ** 2)
    return vmax * hill / (1.0 + alpha_inh * float(row['s_grad'])) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.5, 0.05]), bounds=([0.0, 1e-10, 0.0, 0.0], [100.0, 1e3, 1e3, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))