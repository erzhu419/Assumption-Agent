import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _P
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(p, r) for r in train_rows]) - y
    best = None
    for start in ([3, 2, 0.5, 0.01], [5, 10, 1, 0.1], [2, 0.5, 0.2, 0.001], [10, 20, 2, 0.05]):
        result = least_squares(residual, np.array(start, dtype=float), bounds=([1e-8, 1e-8, 0.0, 0.0], ([100.0, 1e4, 100.0, 10.0])), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _P = best.x.copy()
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_P, row))