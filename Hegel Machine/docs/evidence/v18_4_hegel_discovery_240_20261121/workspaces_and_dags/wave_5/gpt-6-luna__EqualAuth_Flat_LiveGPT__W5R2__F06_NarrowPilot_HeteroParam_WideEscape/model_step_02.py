import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = max(1e-8, float(row['u_now']))
    vmax, km, c = p
    return vmax * u / (km + u) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    p0 = np.array([3.0, 1.0, 0.5])
    res = least_squares(lambda p: np.array([_value(r, p) for r in train_rows]) - y,
                        p0, bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    _params = res.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))