import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 1.0

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    hill = (u * u) / (k_half + u * u)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    initial = np.array([max(1.0, float(np.max(y))), 1.0, 0.5, 0.05])
    result = least_squares(residual, initial, bounds=([1e-9, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_model(row, _PARAMS))