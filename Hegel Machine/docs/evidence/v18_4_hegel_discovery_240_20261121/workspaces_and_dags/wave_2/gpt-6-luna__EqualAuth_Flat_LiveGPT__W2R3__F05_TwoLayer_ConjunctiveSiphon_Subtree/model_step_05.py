import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc = p
    u_n = max(1e-8, float(row['u_now']) / 0.05)
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(u_n)

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.5, 0.70, 0.25]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))