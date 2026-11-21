import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows])
    bounds = ([0.0, 0.0, 0.05, -2.0], [100.0, 30.0, 0.95, 2.0])
    results = []
    for beta in (0.2, 1.0, 3.0, 8.0):
        for c in (-0.5, 0.5):
            results.append(least_squares(residual, [4.0, beta, 0.70, c], bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12))
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_value(row, _params))
