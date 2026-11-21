import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    vmax, rho, c_loc, amp_slow, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u)
    return base - amp_slow * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_evaluate(p, r) - float(r['y']) for r in rows])
    best = None
    for tau in (2.0, 8.0, 20.0, 60.0):
        result = least_squares(residual, [3.0, 0.7, 0.5, 1.0, tau], bounds=([0.0, 0.0, -10.0, 0.0, 0.01], [100.0, 2.0, 10.0, 20.0, 1000.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, row))