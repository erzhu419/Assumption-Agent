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
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 0.5, 0.5, 1.0, 10.0],
        [3.0, 0.2, 1.0, 0.5, 20.0],
        [6.0, 0.8, 0.2, 1.5, 8.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 1e-4, 0.0, 1e-3], [100.0, 1.0, 100.0, 100.0, 500.0]), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda result: float(np.dot(result.fun, result.fun))).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))