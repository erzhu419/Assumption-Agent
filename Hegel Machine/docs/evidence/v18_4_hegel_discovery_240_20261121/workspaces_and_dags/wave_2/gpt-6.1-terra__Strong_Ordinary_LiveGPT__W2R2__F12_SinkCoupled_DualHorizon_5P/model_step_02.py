import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    h = np.asarray([r['H'] for r in rows], dtype=float)
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray [r['y'] for r in train_rows] if False else np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for start in ([8, 0.7, 0.5, 0.8, 10], [12, 0.7, 1, 1.5, 20], [5, 0.7, 0.2, 1, 5]):
        res = least_squares(lambda p: _evaluate(p, train_rows) - y, start, bounds=([0, 0, 0.001, 0, 0.01], [1000, 2, 100, 100, 1000]), max_nfev=5000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x

def predict(row):
    return float(_evaluate(_params, [row])[0])