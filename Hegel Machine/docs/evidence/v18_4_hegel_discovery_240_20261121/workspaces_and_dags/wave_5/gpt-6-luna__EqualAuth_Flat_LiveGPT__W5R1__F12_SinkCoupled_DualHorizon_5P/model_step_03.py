import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    H = float(row['H'])
    s = float(row['s_grad'])
    u = float(row['u_now'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [3.0, 0.5, 0.4, 0.5, 10.0],
        [2.0, 0.2, 0.3, 1.0, 20.0],
        [4.0, 0.8, 0.5, 0.2, 5.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 1e-4, 0.51], [20.0, 1.0, 5.0, 10.0, 200.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))