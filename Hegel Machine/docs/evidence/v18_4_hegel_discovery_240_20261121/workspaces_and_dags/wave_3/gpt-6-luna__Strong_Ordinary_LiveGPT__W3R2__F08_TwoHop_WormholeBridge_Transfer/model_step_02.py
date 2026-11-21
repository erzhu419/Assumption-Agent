import math
import numpy as np
from scipy.optimize import least_squares

def _f(p, row):
    vmax, rho = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_f(p, r) for r in train_rows]) - y
    res = least_squares(residual, x0=np.array([6.0, 0.5]), bounds=([0.01, 0.0], [100.0, 0.99]))
    return res.x

def predict(row):
    global _params
    return float(_f(_params, row))

_params = np.array([6.0, 0.5])
_fit = fit
def fit(train_rows):
    global _params
    _params = _fit(train_rows)
    return _params
