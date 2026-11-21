import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(p, row):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict_with(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(_PARAMS, row))