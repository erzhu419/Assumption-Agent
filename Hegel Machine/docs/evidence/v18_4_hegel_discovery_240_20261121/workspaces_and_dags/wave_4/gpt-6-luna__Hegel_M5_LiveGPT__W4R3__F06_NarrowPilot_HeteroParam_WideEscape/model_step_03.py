import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return u, s

def _value(row, p):
    vmax, rho, c_loc = p
    u, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(
        residual,
        x0=np.array([7.0, 0.5, 0.1]),
        bounds=(np.array([1.0, 0.05, -2.0]), np.array([15.0, 0.95, 2.0])),
    )
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))