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
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    starts = [
        [4.0, 0.3, 0.5, 0.5, 8.0],
        [3.0, 0.5, 0.3, 0.3, 15.0],
        [5.0, 0.2, 1.0, 0.2, 25.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 1e-4, 0.0, 0.05], [30.0, 0.999, 20.0, 10.0, 200.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = best.x
    return _params.tolist()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))