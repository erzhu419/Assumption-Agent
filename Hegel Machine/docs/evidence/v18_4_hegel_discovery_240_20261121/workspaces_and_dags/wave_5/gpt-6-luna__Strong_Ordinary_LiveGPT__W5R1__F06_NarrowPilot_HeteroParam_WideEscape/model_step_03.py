import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(float(row['u_now']), 1e-12)
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [np.array([4.0, 1.0, 0.7, 0.0]), np.array([2.0, 0.2, 0.5, 0.5]), np.array([6.0, 2.0, 0.9, -0.5])]
    fits = [least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 20.0, 2.0, 10.0]), max_nfev=20000) for p0 in starts]
    best = min(fits, key=lambda z: float(np.sum(z.fun ** 2)))
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))
