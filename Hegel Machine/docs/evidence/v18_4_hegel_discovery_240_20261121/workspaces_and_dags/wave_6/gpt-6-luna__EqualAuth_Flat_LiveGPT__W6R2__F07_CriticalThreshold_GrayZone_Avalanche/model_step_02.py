import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    x0 = np.array([5.0, 0.5, 0.5, 2.0, 2.6])
    lower = np.array([0.01, 0.0, 0.0, 0.0, 2.3])
    upper = np.array([30.0, 1.0, 10.0, 20.0, 3.2])
    def residual(p):
        return np.array([_predict_with(p, r) for r in rows]) - y
    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_predict_with(_params, row))