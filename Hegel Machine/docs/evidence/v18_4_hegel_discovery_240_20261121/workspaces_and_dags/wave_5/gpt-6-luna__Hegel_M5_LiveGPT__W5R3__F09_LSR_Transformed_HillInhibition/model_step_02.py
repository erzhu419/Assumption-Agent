import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU_SCALE = 0.1

def _value(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    starts = ([4.0, 1.0, 0.2, 0.01], [2.0, 10.0, 1.0, 0.001], [8.0, 100.0, 0.05, 0.1])
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(_PARAMS, row))