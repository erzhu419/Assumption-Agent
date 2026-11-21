import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    # Optimize log-parameters so amplitudes and time constants remain positive.
    def residual(logp):
        p = np.exp(logp)
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    starts = [np.log([1.0, 0.3, 1.0, 10.0]), np.log([2.0, 0.5, 1.0, 15.0]), np.log([1.0, 1.0, 2.0, 20.0])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(np.log([1e-5, 1e-4, 1e-5, 1e-3]), np.log([100.0, 100.0, 100.0, 1e4])), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = np.exp(best.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _params))