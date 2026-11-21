import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows], dtype=float) - y
    starts = [
        np.array([3.0, 1.0, 0.5, 0.05]),
        np.array([5.0, 2.0, 1.0, 0.1]),
        np.array([2.0, 0.5, 0.1, 0.0]),
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = best.x
    return {'vmax': float(_params[0]), 'k_half': float(_params[1]), 'alpha_inh': float(_params[2]), 'c_leak': float(_params[3])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))