import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.4, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))