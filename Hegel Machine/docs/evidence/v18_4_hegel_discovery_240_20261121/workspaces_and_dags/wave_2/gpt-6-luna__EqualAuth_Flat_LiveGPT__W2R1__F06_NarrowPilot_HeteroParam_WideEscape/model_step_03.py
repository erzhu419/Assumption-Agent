import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    vmax, rho, km, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.8, 1.2, 0.2]), bounds=([0.0, 0.0, 1e-5, 0.0], [100.0, 1.0, 100.0, 10.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict(row, _params))