import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    a, k, c, b = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return (a * u / (k + u) + c * math.sqrt(u)) * math.exp(-b * s)

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=[5.0, 1.0, 0.5, 0.3], bounds=([0.0, 1e-6, 0.0, 0.0], [100.0, 100.0, 100.0, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS.tolist()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _PARAMS))