import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = [[1.0, 1.0, 0.7, 1.0], [2.0, 2.0, 0.5, 0.5], [0.5, 0.2, 0.8, 2.0]]
    sols = [least_squares(residual, p0, bounds=([0.0, 0.0, -2.0, -10.0], [20.0, 20.0, 2.0, 10.0]), max_nfev=20000) for p0 in starts]
    _PARAMS = min(sols, key=lambda z: float(np.sum(z.fun ** 2))).x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _PARAMS))