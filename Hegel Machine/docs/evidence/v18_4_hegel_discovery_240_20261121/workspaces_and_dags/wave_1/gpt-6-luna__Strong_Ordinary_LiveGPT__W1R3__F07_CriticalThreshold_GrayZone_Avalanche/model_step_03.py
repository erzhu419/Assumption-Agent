import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u
    result = least_squares(lambda p: pred(p)-y, x0=np.array([8.0, 2.0, 1.0, 0.5]), bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 20.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k_half, alpha_inh, c_leak = _PARAMS
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u)
