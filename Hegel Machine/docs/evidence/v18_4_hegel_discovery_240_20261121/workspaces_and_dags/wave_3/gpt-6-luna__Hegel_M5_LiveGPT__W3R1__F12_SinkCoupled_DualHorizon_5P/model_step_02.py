import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    lower = np.array([0.01, 0.0, 0.01, 0.0, 0.01])
    upper = np.array([30.0, 0.999, 200.0, 30.0, 500.0])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [6.5, 0.70, 1.0, 0.35, 12.0],
        [5.0, 0.50, 0.5, 0.15, 25.0],
        [8.0, 0.80, 2.0, 0.50, 50.0],
        [4.0, 0.30, 0.2, 0.10, 8.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, np.clip(x0, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=30000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = np.asarray(best[1], dtype=float)
    return {'vmax': float(_params[0]), 'rho': float(_params[1]), 'tau_fast': float(_params[2]), 'amp_adapt': float(_params[3]), 'tau_slow': float(_params[4])}

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))