import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])  # tau_scale = 1
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.70, 0.5]), bounds=([0.0, 0.0, 0.05, 0.0], [100.0, 20.0, 0.95, 20.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(_params, row))