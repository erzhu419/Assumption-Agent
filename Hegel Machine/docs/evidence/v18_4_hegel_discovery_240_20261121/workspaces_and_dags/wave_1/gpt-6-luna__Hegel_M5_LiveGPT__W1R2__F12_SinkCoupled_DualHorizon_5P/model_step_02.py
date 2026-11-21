import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    lower = np.array([0.01, 0.0, 0.0, 0.01, 2.0])
    upper = np.array([30.0, 0.99, 10.0, 5.0, 150.0])
    best = None
    for start in ([6.5, 0.7, 1.0, 0.5, 20.0], [5.0, 0.5, 1.0, 1.0, 15.0], [9.0, 0.75, 2.0, 0.2, 30.0], [4.0, 0.2, 0.5, 2.0, 10.0]):
        result = least_squares(residual, np.asarray(start), bounds=(lower, upper), max_nfev=30000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1].copy()
    return {'vmax': float(_params[0]), 'rho': float(_params[1]), 'amp_adapt': float(_params[2]), 'tau_fast': float(_params[3]), 'tau_slow': float(_params[4])}

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))