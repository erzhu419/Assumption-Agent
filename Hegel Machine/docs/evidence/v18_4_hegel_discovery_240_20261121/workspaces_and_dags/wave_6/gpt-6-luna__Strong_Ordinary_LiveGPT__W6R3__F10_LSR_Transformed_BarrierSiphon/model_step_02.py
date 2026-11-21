import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    root_u = np.sqrt(np.maximum(u, 0.0))
    inhib = 1.0 - np.outer(np.ones_like(u), np.zeros(1)).ravel()  # shape-compatible initialization
    def pred(p):
        j0, beta_bar, rho, c_sqrt = p
        return j0 * u * np.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * root_u
    res = least_squares(lambda p: pred(p) - y, x0=[5.0, 0.5, 0.5, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 10.0, 2.0, np.inf]))
    _params = res.x
    return _params

def predict(row):
    if _params is None:
        raise ValueError('fit must be called before predict')
    j0, beta_bar, rho, c_sqrt = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0)))