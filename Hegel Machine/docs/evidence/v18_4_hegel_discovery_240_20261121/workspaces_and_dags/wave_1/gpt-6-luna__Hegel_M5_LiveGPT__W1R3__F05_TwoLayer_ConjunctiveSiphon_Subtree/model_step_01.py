import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x0 = np.array([6.5, 0.70, 0.1], dtype=float)
    lo = np.array([1.0, 0.05, -2.0])
    hi = np.array([15.0, 0.95, 2.0])
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=10000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))