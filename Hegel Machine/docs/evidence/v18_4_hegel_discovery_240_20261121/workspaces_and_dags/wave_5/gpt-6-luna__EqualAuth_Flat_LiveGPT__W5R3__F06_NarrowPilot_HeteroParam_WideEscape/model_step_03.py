import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    vmax, rho, km, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.2, 0.8, 0.3]), bounds=([0.0, 0.0, 1e-6, 0.0], [100.0, 0.999, 100.0, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _PARAMS))