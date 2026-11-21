import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.2], bounds=([0.0, 0.0, 0.0, -10.0], [50.0, 30.0, 1.0, 10.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))