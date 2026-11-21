import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.70, 0.1]), bounds=([0.0, 0.0, 0.05, -2.0], [30.0, 20.0, 0.95, 2.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict(row, _params))