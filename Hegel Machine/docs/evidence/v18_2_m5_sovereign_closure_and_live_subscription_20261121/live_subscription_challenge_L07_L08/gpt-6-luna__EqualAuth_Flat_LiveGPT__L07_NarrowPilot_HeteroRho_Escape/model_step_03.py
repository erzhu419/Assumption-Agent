import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    vmax, rho, escape = p
    u = float(row['u_now'])  # tau_scale = 1
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    # A fitted additive branch activation captures the observed low-gradient departure.
    return base + escape * max(0.0, 1.0 - s)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.0]),
                           bounds=([0.01, 0.0, -20.0], [100.0, 1.5, 20.0]))
    _PARAMS = result.x
    return result.x

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_predict_with(row, _PARAMS))