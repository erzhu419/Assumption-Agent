import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt, k_lag = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    base = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)
    lag = float(row['H']) * (float(row['mem_conv']) - float(row['x_now']))
    return base + k_lag * lag

def fit(train_rows):
    global _params
    def residual(p):
        return np.asarray([_value(r, p) - float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.5, 0.0]), bounds=([0.0, 0.0, 0.0, 0.0, -10.0], [20.0, 20.0, 1.0, 20.0, 10.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise ValueError('fit must be called before predict')
    return float(_value(row, _params))