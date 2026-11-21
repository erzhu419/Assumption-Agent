import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    result = least_squares(
        residual,
        x0=np.array([5.0, 0.7, 0.0, 2.0, 2.6]),
        bounds=([0.0, 0.0, -20.0, 0.0, 2.0], [100.0, 1.0, 20.0, 20.0, 3.5]),
        max_nfev=20000
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))