import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _base(row, p):
    vmax, rho, bx, bm, bh = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    basin = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    leak = bx * float(row['x_now']) + bm * float(row['mem_conv']) + bh * (float(row['u_past']) - u)
    return basin + leak

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_base(r, p) for r in rows]) - y
    # Basin scale and suppression initialized from the recalled mechanism;
    # external-leak coefficients start at zero.
    result = least_squares(residual, np.array([2.5, 0.70, 0.0, 0.0, 0.0]),
                           bounds=([0.01, 0.0, -5.0, -5.0, -5.0],
                                   [20.0, 1.5, 5.0, 5.0, 5.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_base(row, _PARAMS))