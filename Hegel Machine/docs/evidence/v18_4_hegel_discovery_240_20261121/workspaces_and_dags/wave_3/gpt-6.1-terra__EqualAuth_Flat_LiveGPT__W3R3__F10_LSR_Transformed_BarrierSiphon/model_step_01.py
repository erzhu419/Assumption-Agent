import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = row['u_now'] / 0.02
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for beta in (0.1, 1.0, 3.0, 8.0):
        result = least_squares(residual, [3.0, beta, 0.5, 0.5], bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 30.0, 1.0, np.inf]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))