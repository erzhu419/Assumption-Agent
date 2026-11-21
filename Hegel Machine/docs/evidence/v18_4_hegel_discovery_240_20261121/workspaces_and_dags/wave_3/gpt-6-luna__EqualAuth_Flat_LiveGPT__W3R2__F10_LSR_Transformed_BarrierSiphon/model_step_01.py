import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 3.0
_params = None

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 1.0, 0.3, 0.5]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 20.0, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _params))