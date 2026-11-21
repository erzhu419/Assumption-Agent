import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = max(0.0, float(row['H']))
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    if len(rows) < 4:
        raise ValueError('Insufficient rows for the four-parameter dual-horizon model')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    # Optimize log-parameters to keep amplitudes and time constants positive.
    def residual(logp):
        p = np.exp(logp)
        return np.asarray([_response(r, p) for r in rows]) - y
    starts = [np.log([0.8, 0.3, 0.8, 15.0]), np.log([1.0, 0.5, 1.0, 10.0]), np.log([0.5, 0.2, 0.5, 25.0])]
    results = [least_squares(residual, s, max_nfev=20000) for s in starts]
    best = min(results, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = np.exp(best.x)
    return _params

def predict(row):
    if _params is None:
        raise ValueError('Call fit(train_rows) before predict(row)')
    return float(_response(row, _params))