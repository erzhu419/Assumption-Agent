import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(z):
        p = np.exp(z)
        return np.asarray([_response(r, p) for r in train_rows]) - ys
    starts = [np.log([1.5, 0.3, 0.5, 10.0]), np.log([2.0, 0.5, 1.0, 20.0]), np.log([3.0, 1.0, 0.5, 8.0])]
    fits = [least_squares(residual, s, max_nfev=30000) for s in starts]
    best = min(fits, key=lambda q: float(np.sum(q.fun ** 2)))
    _params = np.exp(best.x)
    return _params.tolist()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _params))