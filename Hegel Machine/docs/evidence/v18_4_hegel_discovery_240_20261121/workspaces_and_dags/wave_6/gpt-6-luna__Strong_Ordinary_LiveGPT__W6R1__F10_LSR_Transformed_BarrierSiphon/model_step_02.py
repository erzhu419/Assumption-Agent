import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - X
    starts = [np.array([2.0, 1.0, 0.5, 0.1]), np.array([4.0, 2.0, 0.7, 0.0]), np.array([1.0, 0.2, 0.2, 0.5])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 1.0, np.inf]), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))