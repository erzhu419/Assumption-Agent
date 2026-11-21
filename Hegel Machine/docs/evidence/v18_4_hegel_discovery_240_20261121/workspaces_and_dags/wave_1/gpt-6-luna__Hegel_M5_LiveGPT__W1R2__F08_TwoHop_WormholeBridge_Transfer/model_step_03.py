import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c * math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    result = least_squares(
        residual,
        x0=np.array([8.0, 0.70, 0.1]),
        bounds=(np.array([1.0, 0.05, -2.0]), np.array([15.0, 0.95, 2.0])),
    )
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _params))