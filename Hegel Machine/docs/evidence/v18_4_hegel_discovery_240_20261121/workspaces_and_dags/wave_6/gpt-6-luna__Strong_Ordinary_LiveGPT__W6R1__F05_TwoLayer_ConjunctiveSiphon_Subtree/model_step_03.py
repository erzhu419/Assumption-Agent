import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    base = (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return vmax * base + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.1]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _params))