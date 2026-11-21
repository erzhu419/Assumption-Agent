import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    lower = np.array([0.0, 0.0, 0.0, 1e-4, 1e-4])
    upper = np.array([100.0, 1.0, 100.0, 100.0, 1000.0])
    starts = [np.array([4.0, 0.3, 0.5, 0.5, 12.0]),
              np.array([3.0, 0.6, 0.4, 1.0, 25.0]),
              np.array([6.0, 0.2, 0.8, 0.2, 8.0])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            x0, bounds=(lower, upper), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))