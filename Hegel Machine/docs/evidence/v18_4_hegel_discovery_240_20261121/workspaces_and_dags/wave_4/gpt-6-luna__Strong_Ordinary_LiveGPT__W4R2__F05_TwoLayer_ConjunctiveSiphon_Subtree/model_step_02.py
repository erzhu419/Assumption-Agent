import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.05

def _predict_with(p, row):
    vmax, rho, c_loc, km = p
    u_n = max(0.0, float(row['u_now']) / TAU_SCALE)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (km + u_n))
    return base + c_loc * math.sqrt(u_n)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=[5.0, 0.5, 0.5, 1.2],
                           bounds=([0.0, 0.0, -20.0, 1e-6], [100.0, 1.0, 20.0, 20.0]),
                           max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(_PARAMS, row))