import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    j, beta, rho, c = p
    return j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [float(r['y']) for r in rows]
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for beta in (0.1, 0.5, 1.0, 2.0, 4.0):
        result = least_squares(residual, [2.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 1.0, np.inf]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row) -> float:
    return float(_value(row, _params))
