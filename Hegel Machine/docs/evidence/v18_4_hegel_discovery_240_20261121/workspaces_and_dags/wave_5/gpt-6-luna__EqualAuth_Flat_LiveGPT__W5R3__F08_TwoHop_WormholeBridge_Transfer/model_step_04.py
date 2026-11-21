import math
import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def _pred(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic_tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _fit_params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_pred(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]))
    _fit_params = result.x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _fit_params))