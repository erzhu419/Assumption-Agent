import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return u, s

def _value(params, row):
    vmax, k_half, alpha_inh, c_leak = params
    u, s = _features(row)
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(fit(_TRAIN_ROWS), row))