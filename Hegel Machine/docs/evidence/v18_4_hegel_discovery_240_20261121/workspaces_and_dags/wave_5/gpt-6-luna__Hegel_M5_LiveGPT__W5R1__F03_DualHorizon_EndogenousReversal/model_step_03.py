import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    H = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_response(r, p) for r in train_rows]) - ys
    starts = [
        [1.5, 0.4, 1.2, 15.0],
        [2.0, 0.5, 1.5, 20.0],
        [1.0, 0.2, 1.0, 10.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 1e-5, 0.0, 1e-5], [100.0, 1e4, 100.0, 1e4]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.sum(result.fun ** 2)))
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _params))