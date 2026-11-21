import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.02
_PARAMS = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[6.0, 1.0, 0.70, 0.1], bounds=([0.0, 0.0, 0.05, 0.0], [1000.0, 100.0, 0.95, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))