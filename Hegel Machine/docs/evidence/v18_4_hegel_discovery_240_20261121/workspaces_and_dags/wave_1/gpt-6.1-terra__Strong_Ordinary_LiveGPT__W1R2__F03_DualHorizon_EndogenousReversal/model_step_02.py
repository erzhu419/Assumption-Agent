import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, h):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _evaluate(p, u, h) - y
    best = None
    for start in ([1.5, 2.0, 0.3, 10.0], [2.0, 3.0, 1.0, 15.0], [1.0, 2.0, 0.1, 5.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.001, 0.01], [100.0, 100.0, 100.0, 1000.0]), max_nfev=10000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['H'])))