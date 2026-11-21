import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return u, s, h

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast))
    slow = amp_adapt * u * (-np.expm1(-h / tau_slow))
    return fast - slow

def fit(train_rows):
    global _params
    u, s, h = _values(train_rows)
    y = np.array [r['y'] for r in train_rows]
    y = np.asarray(y, dtype=float)
    best = None
    for start in ([5, 0.7, 0.5, 0.6, 10], [10, 0.7, 1, 1, 20], [4, 0.5, 0.2, 0.8, 5]):
        result = least_squares(lambda p: _model(p, u, s, h) - y, start, bounds=([0, 0, 0.001, 0, 0.01], [1000, 1, 100, 100, 1000]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    u, s, h = _values([row])
    return float(_model(_params, u, s, h)[0])