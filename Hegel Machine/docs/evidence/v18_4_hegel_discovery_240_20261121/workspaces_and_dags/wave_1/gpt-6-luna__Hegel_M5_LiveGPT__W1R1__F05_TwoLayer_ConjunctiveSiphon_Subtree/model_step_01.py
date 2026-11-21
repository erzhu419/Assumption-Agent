import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 4.0

def _predict_params(row, p):
    vmax, rho, c_loc = p
    u_n = max(0.0, float(row['u_now'])) / _TAU
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(u_n)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]),
                           bounds=(np.array([1.0, 0.05, -2.0]),
                                   np.array([15.0, 0.95, 2.0])))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _PARAMS))