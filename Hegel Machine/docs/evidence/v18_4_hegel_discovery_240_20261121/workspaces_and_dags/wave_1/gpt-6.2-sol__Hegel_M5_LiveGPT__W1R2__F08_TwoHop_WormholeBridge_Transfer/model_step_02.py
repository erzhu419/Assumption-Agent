import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, drive, grad, root):
    vmax, rho, c = p
    return vmax * (1.0 - rho * grad) * drive + c * root

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    drive = u / (1.2 + u)
    grad = s / (1.0 + s)
    root = np.sqrt(u)
    _params = least_squares(
        lambda p: _response(p, drive, grad, root) - y,
        x0=[6.5, 0.5, 0.0],
        bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]),
    ).x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_response(_params, u / (1.2 + u), s / (1.0 + s), math.sqrt(u)))
