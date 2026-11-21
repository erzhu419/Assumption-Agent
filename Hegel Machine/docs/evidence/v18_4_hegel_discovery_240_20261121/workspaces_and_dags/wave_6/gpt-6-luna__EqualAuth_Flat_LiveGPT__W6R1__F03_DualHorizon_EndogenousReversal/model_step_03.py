import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    s = float(row['s_grad'])
    sink = 1.0  # s_grad is zero in every observed row; sink coupling is unidentifiable.
    fast = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    slow = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return fast - slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=[2.0, 0.5, 0.1, 10.0], bounds=([0.0, 1e-6, 0.0, 1e-6], [100.0, 1000.0, 100.0, 1000.0]), max_nfev=50000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))