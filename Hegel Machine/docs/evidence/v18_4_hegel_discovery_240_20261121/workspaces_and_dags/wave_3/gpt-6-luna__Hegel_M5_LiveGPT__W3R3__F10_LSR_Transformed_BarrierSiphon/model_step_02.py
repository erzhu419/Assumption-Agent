import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.02

def _features(row):
    u = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return u, s

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.70, 0.10]), bounds=([0.0, 0.0, 0.05, -10.0], [100.0, 20.0, 0.95, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _PARAMS))
