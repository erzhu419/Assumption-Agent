import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    inhibition = 1.0 - rho * s / (1.0 + s)
    base = vmax * inhibition * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    lower = np.array([0.0, 0.0, 0.01, 0.0, 0.05])
    upper = np.array([20.0, 1.0, 20.0, 5.0, 200.0])
    starts = [
        [3.0, 0.5, 0.3, 0.5, 8.0],
        [4.0, 0.7, 1.0, 0.3, 15.0],
        [2.0, 0.3, 0.1, 0.8, 5.0],
        [5.0, 0.8, 2.0, 0.2, 30.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, np.asarray(x0, dtype=float), bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = np.asarray(best[1], dtype=float)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))