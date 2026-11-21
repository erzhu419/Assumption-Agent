import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    return u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_pred(r, p) for r in rows]) - y
    best = None
    # Positive amplitudes and timescales; multistart helps resolve the nonlinear time constants.
    for tf0, ts0 in [(0.2, 5.0), (0.5, 10.0), (1.0, 20.0), (2.0, 50.0), (0.1, 30.0)]:
        p0 = np.array([0.8, tf0, 0.8, ts0])
        res = least_squares(residual, p0, bounds=([0.0, 1e-4, 0.0, 1e-4], [20.0, 1e3, 20.0, 1e3]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    return float(_pred(row, _FITTED_PARAMS))

_FITTED_PARAMS = None

def _set_model(train_rows):
    global _FITTED_PARAMS
    _FITTED_PARAMS = fit(train_rows)
    return _FITTED_PARAMS
