import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def predict_params(p, row):
        vmax, rho = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        un = u  # tau_scale is 1.0
        return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    def residual(p):
        return np.array([predict_params(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([8.0, 0.7]), bounds=([0.0, 0.0], [np.inf, 1.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, rho = _PARAMS
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)))