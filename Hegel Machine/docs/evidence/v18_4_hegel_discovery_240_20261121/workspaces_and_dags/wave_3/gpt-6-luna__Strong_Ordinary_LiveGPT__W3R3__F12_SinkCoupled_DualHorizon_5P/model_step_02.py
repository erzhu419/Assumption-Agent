import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in rows], dtype=float) - y
    lower = np.array([0.0, 0.0, 1e-4, 0.0, 1e-3])
    upper = np.array([20.0, 1.0, 20.0, 5.0, 200.0])
    starts = [np.array([3.0, 0.5, 0.5, 0.8, 10.0]), np.array([5.0, 0.7, 1.0, 1.0, 20.0]), np.array([2.0, 0.3, 0.2, 0.5, 5.0])]
    results = [least_squares(residual, np.clip(p, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=20000) for p in starts]
    best = min(results, key=lambda res: float(np.dot(res.fun, res.fun)))
    _PARAMS = best.x.copy()
    return {'vmax': float(_PARAMS[0]), 'rho': float(_PARAMS[1]), 'tau_fast': float(_PARAMS[2]), 'amp_adapt': float(_PARAMS[3]), 'tau_slow': float(_PARAMS[4])}

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(_PARAMS, row))