import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + branch * x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(
        residual,
        x0=np.array([7.0, 0.2, 0.0]),
        bounds=(np.array([1.0, 0.05, -2.0]), np.array([15.0, 0.95, 2.0]))
    )
    _PARAMS = result.x
    return result.x

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))