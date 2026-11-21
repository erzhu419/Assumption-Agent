import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    vmax, km, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = vmax * (u / (km + u))
    sink = 1.0 - rho * (s / (1.0 + s))
    return base * sink + c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.1, 0.2, 0.5]), bounds=([0.0, 1e-6, 0.0, 0.0], [100.0, 10.0, 1.0, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_predict_with(row, _PARAMS))