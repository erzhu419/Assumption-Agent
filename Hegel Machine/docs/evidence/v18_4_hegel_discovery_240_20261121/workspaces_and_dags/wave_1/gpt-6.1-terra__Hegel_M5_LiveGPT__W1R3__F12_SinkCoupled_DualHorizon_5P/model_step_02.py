import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([0.01, 0.0, 0.001, 0.0, 0.05], [50.0, 1.5, 20.0, 20.0, 500.0])
    starts = [[6.5, 0.7, 0.5, 1.5, 10.0], [6.5, 0.7, 1.0, 2.0, 20.0], [6.5, 0.7, 0.3, 1.0, 5.0]]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))