import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = row['u_now'] / 0.02
    s = row['s_grad']
    j0, beta, rho, c = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray [float(r['y']) for r in train_rows]
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    for beta in (0.5, 2.0, 5.0):
        for c in (-0.5, 0.5):
            result = least_squares(residual, [5.0, beta, 0.7, c], bounds=([0.0, 0.0, 0.05, -10.0], [100.0, 30.0, 0.95, 10.0]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))