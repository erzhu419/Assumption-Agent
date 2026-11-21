import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _calc(row, p):
    vmax, rho, c_loc = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.array([_calc(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=[6.0, 0.70, 0.1], bounds=([3.836395, 0.658333, -0.822167], [9.163605, 0.748333, 1.108834]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise ValueError('fit(train_rows) must be called first')
    return float(_calc(row, _PARAMS))