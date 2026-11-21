import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _law(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_law(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=[5.0, 0.5, 0.5, 1.0, 2.6],
                           bounds=([0.0, 0.0, 0.0, 0.0, 2.0],
                                   [100.0, 10.0, 20.0, 20.0, 3.2]),
                           max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_law(row, _PARAMS))