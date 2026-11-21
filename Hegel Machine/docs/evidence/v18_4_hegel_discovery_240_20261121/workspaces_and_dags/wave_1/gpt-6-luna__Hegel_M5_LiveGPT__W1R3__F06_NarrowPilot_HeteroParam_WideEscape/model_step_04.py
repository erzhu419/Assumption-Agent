import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, p):
    j0, beta, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    best = None
    for start in ([5.0, 1.0, 0.5, 0.1], [10.0, 2.0, 0.3, 0.5], [3.0, 0.2, 0.8, 1.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 30.0, 0.999, 10.0]), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _PARAMS))