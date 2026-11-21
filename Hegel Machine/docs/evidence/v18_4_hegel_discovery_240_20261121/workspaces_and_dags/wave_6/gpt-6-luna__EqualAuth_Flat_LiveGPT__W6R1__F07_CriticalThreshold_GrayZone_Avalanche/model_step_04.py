import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_law(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.5, 0.2, 1.5, 2.6]), max_nfev=20000)
    _params = result.x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_law(row, _params))