import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    starts = ([6.0, 2.0, 0.2, 0.05], [10.0, 5.0, 0.5, 0.01], [5.0, 0.5, 0.05, 0.1])
    for start in starts:
        res = least_squares(residual, x0=np.asarray(start), bounds=([0.0, 1e-9, 0.0, 0.0], [1000.0, 1e4, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))