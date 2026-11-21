import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, rows):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast))
    slow = amp_adapt * u * (-np.expm1(-h / tau_slow))
    return fast - slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray [r['y'] for r in rows], dtype=float)
    best = None
    for tf in (0.1, 0.3, 0.8, 2.0):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(lambda p: _model(p, rows) - y, [5.0, 0.7, tf, 1.0, ts], bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 1.5, 100.0, 100.0, 1000.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, [row])[0])