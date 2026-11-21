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
    def residual(p):
        return np.array([_value(row, p) - row['y'] for row in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        for c in (0.0, 1.0):
            result = least_squares(residual, [6.5, beta, 0.70, c], bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 30.0, 1.0, 10.0]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))