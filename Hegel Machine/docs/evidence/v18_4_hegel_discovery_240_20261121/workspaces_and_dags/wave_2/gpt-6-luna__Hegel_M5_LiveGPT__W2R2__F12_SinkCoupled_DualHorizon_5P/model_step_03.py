import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [6.0, 0.70, 0.5, 0.8, 12.0],
        [6.0, 0.70, 1.0, 0.8, 20.0],
        [5.0, 0.65, 0.2, 0.6, 8.0],
    ]
    lo = [0.01, 0.0, 0.01, 0.0, 0.02]
    hi = [30.0, 0.999, 10.0, 10.0, 200.0]
    sols = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    sol = min(sols, key=lambda z: float(np.dot(z.fun, z.fun)))
    _PARAMS = np.asarray(sol.x, dtype=float)
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))