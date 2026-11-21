import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    A, K, C, rho, gamma = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    baseline = A * u / (K + u) + C * math.sqrt(u)
    return baseline * (1.0 - rho * s / (1.0 + s)) - gamma * max(0.0, s - 2.0) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 1.0, 0.05, 0.05]), bounds=([0.0, 1e-4, 0.0, -1.0, 0.0], [30.0, 20.0, 10.0, 1.0, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))