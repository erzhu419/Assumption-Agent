import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def pred(p, row):
        u = max(0.0, float(row['u_now']))
        s = float(row['s_grad'])
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.3]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))