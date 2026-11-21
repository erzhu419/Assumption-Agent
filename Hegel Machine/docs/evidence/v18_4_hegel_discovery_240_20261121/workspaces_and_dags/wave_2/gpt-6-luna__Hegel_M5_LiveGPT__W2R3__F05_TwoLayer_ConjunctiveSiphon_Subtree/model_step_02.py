import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    x0 = np.asarray([6.5, 0.70, 0.14], dtype=float)
    lo = np.asarray([3.836397, 0.658333, -0.822167], dtype=float)
    hi = np.asarray([9.163604, 0.748333, 1.108834], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows], dtype=float) - y
    result = least_squares(residual, x0=np.clip(x0, lo, hi), bounds=(lo, hi))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))