import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / 0.02, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    q = s / (1.0 + s)
    root = np.sqrt(u)
    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * q) + c * root - y
    bounds = ([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf])
    fits = [least_squares(residual, [3.0, beta, 0.7, 0.5], bounds=bounds, x_scale='jac') for beta in (0.5, 2.0, 5.0)]
    _params = min(fits, key=lambda result: result.cost).x

def predict(row) -> float:
    j0, beta, rho, c = _params
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))