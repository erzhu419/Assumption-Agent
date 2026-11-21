import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 3.0

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(float(row['u_now']) / _TAU_SCALE, 0.0)
    s = float(row['s_grad'])
    flux = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c_sqrt * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = [np.array([2.0, 1.0, 0.7, 0.0]), np.array([4.0, 2.0, 0.5, 0.1]), np.array([1.0, 0.2, 0.9, -0.1])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_value(row, fit.__dict__.get('_params', [])))