import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(p, row):
    vmax, rho, c_loc = p
    u = float(row['u_now'])  # characteristic_tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.5]), bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict(_params, row))