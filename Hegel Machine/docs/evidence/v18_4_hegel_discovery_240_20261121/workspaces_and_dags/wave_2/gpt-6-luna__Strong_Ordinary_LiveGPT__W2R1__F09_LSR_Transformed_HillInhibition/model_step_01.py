import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 2.5
_params = None

def _pred(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    un = float(row['u_now']) / _TAU
    sg = float(row['s_grad'])
    return vmax * (un * un / (k_half + un * un)) / (1.0 + alpha_inh * sg) - c_leak * un

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(_params, row))