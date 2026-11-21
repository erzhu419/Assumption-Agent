import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_evaluate(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, [6.5, 0.7, 1.0, 1.0, 10.0], bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [30.0, 1.0, 20.0, 10.0, 200.0]), max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_evaluate(row, _params))