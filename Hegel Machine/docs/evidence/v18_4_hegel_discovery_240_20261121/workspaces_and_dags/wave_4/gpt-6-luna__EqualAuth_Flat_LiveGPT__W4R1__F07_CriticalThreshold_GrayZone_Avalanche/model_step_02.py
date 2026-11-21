import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(0.0, u))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(
        residual,
        x0=np.array([4.0, 0.3, 0.5, 1.0, 2.6]),
        bounds=(np.array([0.0, 0.0, 0.0, 0.0, 2.3]),
                np.array([20.0, 1.0, 5.0, 10.0, 2.9])),
        max_nfev=20000,
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))