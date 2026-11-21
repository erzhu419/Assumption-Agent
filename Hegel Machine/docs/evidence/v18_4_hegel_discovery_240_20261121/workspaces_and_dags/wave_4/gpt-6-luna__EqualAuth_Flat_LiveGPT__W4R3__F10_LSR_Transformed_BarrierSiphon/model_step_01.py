import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.02

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    inhibition = 1.0 - rho * s / (1.0 + s)
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * inhibition + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.3, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))