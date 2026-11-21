import numpy as np
from scipy.optimize import least_squares

_params = None

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_params[0] * float(row['u_now']) + _params[1])

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, np.array([2.2, 0.0]))
    _params = result.x
    return _params.copy()
