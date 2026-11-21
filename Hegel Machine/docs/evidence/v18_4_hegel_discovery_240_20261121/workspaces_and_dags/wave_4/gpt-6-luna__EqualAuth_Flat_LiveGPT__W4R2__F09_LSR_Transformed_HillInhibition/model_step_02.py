import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    hill = (u * u) / (k_half + u * u)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_response(r, p) for r in train_rows], dtype=float) - ys
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _params))