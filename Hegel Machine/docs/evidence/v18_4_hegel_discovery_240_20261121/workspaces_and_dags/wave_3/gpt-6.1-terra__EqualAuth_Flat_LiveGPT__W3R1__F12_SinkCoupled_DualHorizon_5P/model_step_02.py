import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_evaluate(p, row) - float(row['y']) for row in train_rows])
    best = None
    for start in ([8.0, 0.5, 0.5, 1.0, 10.0], [12.0, 0.8, 1.0, 1.5, 20.0], [5.0, 0.3, 0.2, 0.5, 5.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-5, 0.0, 1e-5], [np.inf, 1.0, np.inf, np.inf, np.inf]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, row))