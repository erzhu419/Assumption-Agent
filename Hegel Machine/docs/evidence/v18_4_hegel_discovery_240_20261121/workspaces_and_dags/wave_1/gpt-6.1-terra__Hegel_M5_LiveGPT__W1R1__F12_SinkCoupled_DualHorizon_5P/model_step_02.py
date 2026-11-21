import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.1], [30.0, 1.0, 10.0, 10.0, 200.0])
    best = None
    for tf in (0.1, 0.4, 1.0):
        for ts in (3.0, 10.0, 30.0):
            result = least_squares(residual, [6.5, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))