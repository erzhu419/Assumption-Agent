import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = max(0.0, float(row['H']))
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(q):
        # Positive amplitudes and timescales; tau_slow is constrained above tau_fast.
        af, ass, tf, gap = np.exp(q)
        p = (af, ass, tf, tf + gap)
        return np.asarray([_value(r, p) for r in rows]) - y
    best = None
    for af in (0.5, 1.0, 2.0):
        for ass in (0.5, 1.0, 2.0):
            q0 = np.log([af, ass, 0.5, 5.0])
            res = least_squares(residual, q0, max_nfev=20000)
            if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
                best = res
    af, ass, tf, gap = np.exp(best.x)
    _PARAMS = (float(af), float(ass), float(tf), float(tf + gap))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))