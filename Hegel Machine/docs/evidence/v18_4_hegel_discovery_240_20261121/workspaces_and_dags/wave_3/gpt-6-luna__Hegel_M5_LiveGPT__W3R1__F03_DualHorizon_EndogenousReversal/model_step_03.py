import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    h = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def residual(logp):
        amp_fast, amp_slow, tau_fast, tau_slow = np.exp(logp)
        pred = u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))
        return pred - y

    starts = [
        np.log([1.5, 1.0, 0.5, 10.0]),
        np.log([2.0, 1.5, 0.3, 20.0]),
        np.log([1.0, 0.5, 1.0, 8.0]),
    ]
    fits = [least_squares(residual, s, max_nfev=20000) for s in starts]
    best = min(fits, key=lambda f: float(np.dot(f.fun, f.fun)))
    _PARAMS = np.exp(best.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, amp_slow, tau_fast, tau_slow = _PARAMS
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow))))