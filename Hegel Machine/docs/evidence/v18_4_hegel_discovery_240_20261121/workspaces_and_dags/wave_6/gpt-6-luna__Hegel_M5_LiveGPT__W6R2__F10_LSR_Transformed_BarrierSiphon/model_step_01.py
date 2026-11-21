import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [np.array([5.0, 1.0, 0.70, 0.5]), np.array([2.0, 2.0, 0.5, 1.0])]
    bounds = ([0.0, 0.0, 0.05, 0.0], [100.0, 50.0, 0.95, 100.0])
    fits = [least_squares(residual, x0, bounds=bounds, max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda z: np.sum(z.fun ** 2)).x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))