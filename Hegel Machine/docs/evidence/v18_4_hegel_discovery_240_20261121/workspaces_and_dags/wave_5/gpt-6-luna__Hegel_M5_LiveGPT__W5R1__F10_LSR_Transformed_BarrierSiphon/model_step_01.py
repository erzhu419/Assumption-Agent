import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.70, 0.1]), bounds=([0.0, 0.0, 0.05, 0.0], [np.inf, np.inf, 0.95, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_predict_params(row, _params))