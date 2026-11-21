import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    un = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return un, s

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    un, s = _features(row)
    return vmax * (un * un / (k_half + un * un)) / (1.0 + alpha_inh * s) - c_leak * un

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.2, 0.0]), bounds=([0.0, 1e-8, 0.0, -100.0], [1e4, 1e4, 1e3, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    params = fit([row]) if 'y' in row else _PARAMS
    return float(_model(row, params))