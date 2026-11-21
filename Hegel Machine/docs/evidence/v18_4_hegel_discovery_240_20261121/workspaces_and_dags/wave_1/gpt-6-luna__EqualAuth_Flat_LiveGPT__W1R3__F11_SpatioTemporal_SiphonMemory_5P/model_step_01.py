import numpy as np
from scipy.optimize import least_squares
import math

_params = None

def _law(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    X = list(train_rows)
    y = np.array([float(r['y']) for r in X])
    def residual(p):
        return np.array([_law(r, p) for r in X]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.2]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 20.0, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_law(row, _params))