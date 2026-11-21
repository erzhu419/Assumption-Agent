import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _raw(row, p):
    vmax, rho, offset = p
    u_n = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + offset

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    p0 = np.array([6.0, 0.70, 0.0])
    def residual(p):
        return np.asarray([_raw(r, p) for r in train_rows]) - y
    result = least_squares(residual, p0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_raw(row, _PARAMS))