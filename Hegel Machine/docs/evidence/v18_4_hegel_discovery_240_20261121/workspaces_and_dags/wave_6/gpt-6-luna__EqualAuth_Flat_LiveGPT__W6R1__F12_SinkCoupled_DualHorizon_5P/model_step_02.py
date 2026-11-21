import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    H = float(row['H'])
    s = float(row['s_grad'])
    u = float(row['u_now'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    # Positive scales and bounded sink fraction; multistart limits local-fit sensitivity.
    starts = [
        [4.0, 0.4, 0.3, 0.15, 12.0],
        [3.0, 0.6, 0.5, 0.2, 20.0],
        [5.0, 0.2, 0.2, 0.1, 8.0],
    ]
    lo = [0.0, 0.0, 1e-5, 0.0, 1e-4]
    hi = [20.0, 1.0, 100.0, 10.0, 1000.0]
    fits = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda result: float(np.dot(result.fun, result.fun))).x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))