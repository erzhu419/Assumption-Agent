import math
import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def fit(train_rows):
    global _fit_params
    rows = list(train_rows)
    def residuals(params):
        vmax, c_ctrl = params
        return np.array([vmax * float(r['x_ind']) / (1.1 + float(r['x_ind'])) + c_ctrl * float(r['u_ctrl']) - float(r['y']) for r in rows])
    result = least_squares(residuals, x0=np.array([5.0, 0.5]))
    _fit_params = result.x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, c_ctrl = _fit_params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
