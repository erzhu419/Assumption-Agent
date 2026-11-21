import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([5, 0.5, 0.5, 1, 10], [10, 0.8, 1, 1, 20], [5, 1, 0.2, 0.5, 5]):
        result = least_squares(residual, start, bounds=([0, 0, 0.001, 0, 0.01], [100, 2, 100, 100, 1000]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))