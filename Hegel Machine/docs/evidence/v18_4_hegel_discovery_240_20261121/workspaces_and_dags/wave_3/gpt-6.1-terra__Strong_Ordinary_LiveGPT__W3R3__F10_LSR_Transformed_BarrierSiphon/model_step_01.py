import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _eval(row, p):
    j0, beta, rho, c = p
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_eval(row, p) - row['y'] for row in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 30.0, 1.0, 20.0]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_eval(row, _params))