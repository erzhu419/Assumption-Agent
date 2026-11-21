import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.5, 0.7, 0.0]), np.array([5.0, 1.0, 0.5, 0.2]), np.array([1.0, 0.1, 0.9, -0.2])]
    fits = [least_squares(residual, x0, max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: np.sum(z.fun ** 2))
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _params))