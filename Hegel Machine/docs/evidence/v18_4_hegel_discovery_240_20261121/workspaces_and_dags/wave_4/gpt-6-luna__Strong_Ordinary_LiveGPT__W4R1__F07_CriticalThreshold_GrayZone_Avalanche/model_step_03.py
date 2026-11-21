import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(
        residual,
        x0=np.array([5.0, 0.7, 0.0, 1.0, 2.6]),
        bounds=(np.array([0.0, 0.0, -10.0, 0.0, 2.0]),
                np.array([20.0, 1.0, 10.0, 10.0, 3.2])),
        max_nfev=20000
    )
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))