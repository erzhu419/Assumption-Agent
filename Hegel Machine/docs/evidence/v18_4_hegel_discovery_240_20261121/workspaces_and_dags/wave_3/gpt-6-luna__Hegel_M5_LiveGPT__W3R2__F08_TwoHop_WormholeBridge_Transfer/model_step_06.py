import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def pred(p, r):
        vmax, rho, c_loc = p
        u = max(1e-9, float(r['u_now']))
        s = float(r['s_grad'])
        siphon = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return siphon + c_loc * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c_loc = _params
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))
