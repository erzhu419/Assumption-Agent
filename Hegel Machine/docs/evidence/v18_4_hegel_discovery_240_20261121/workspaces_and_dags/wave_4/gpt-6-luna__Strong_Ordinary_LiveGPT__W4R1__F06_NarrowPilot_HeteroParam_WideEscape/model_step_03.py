import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic_tau_scale = 1.0
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_law(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([10.0, 1.0, 0.5, 0.1]), bounds=([0.0, 0.0, -5.0, -5.0], [100.0, 20.0, 5.0, 5.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_law(row, _params))