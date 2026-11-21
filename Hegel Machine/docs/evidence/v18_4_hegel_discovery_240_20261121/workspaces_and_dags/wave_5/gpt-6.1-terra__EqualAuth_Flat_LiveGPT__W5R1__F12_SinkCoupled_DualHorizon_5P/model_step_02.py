import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast))
    slow = amp_adapt * u * (-np.expm1(-h / tau_slow))
    return fast - slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_evaluate(r, p) for r in rows]) - y
    best = None
    for start in ([5.0, 0.5, 0.5, 1.0, 8.0], [8.0, 1.0, 1.0, 2.0, 15.0], [3.0, 0.2, 0.2, 0.5, 4.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 2.0, 100.0, 100.0, 1000.0]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(row, _params))