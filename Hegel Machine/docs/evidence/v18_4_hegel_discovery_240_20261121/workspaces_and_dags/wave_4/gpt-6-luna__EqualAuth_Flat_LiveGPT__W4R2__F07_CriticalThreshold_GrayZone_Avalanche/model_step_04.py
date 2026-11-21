import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.5, 0.5, 1.0, 2.6], [5.0, 0.8, 0.2, 1.5, 2.5], [3.0, 0.3, 1.0, 0.8, 2.7]):
        result = least_squares(residual, start,
            bounds=([0.0, 0.0, -10.0, 0.0, 0.0], [100.0, 1.0, 10.0, 20.0, 10.0]),
            max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))