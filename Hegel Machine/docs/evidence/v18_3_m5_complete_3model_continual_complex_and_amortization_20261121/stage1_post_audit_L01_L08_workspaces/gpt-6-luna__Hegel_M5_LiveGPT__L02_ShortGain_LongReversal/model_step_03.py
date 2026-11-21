import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    return float(row['u_now']) * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    def unpack(z):
        return (math.exp(z[0]), math.exp(z[1]), math.exp(z[2]), math.exp(z[3]))
    def residual(z):
        p = unpack(z)
        return np.asarray([_model(r, p) - float(r['y']) for r in train_rows], dtype=float)
    starts = [np.log([1.0, 1.0, 0.5, 10.0]), np.log([2.0, 1.0, 0.2, 20.0]), np.log([1.0, 2.0, 1.0, 15.0])]
    results = [least_squares(residual, s, max_nfev=30000) for s in starts]
    best = min(results, key=lambda r: float(np.dot(r.fun, r.fun)))
    _PARAMS = unpack(best.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _model(row, _PARAMS)