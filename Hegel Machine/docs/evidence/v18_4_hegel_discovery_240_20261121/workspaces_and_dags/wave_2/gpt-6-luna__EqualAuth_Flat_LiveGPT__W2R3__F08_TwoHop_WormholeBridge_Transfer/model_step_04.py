import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    a, k, c, d = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return c + a * u / (k + max(u, 0.0)) + d * (s - 1.0)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    p0 = np.array([3.0, 1.0, 0.0, -0.8])
    res = least_squares(lambda p: np.array([_value(p, r) for r in train_rows]) - y,
                        p0, max_nfev=20000)
    _PARAMS = res.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(_PARAMS, row))