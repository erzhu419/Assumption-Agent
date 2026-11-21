import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    vmax, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    fast = vmax * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    slow = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return fast - slow

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5, 15.0]), bounds=([0.0, 1e-4, 0.0, 0.5001], [100.0, 20.0, 100.0, 1e4]), max_nfev=20000)
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _P))