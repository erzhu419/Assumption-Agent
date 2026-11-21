import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.2]),
                           bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))