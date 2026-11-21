import math
import numpy as np
from scipy.optimize import least_squares
_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def predict_values(p):
        vmax, k, n, c = p
        xn = np.maximum(x, 1e-12) ** n
        kn = k ** n
        return vmax * xn / (kn + xn) + c * u

    result = least_squares(lambda p: predict_values(p) - y,
                           x0=np.array([8.0, 2.0, 2.0, 0.5]),
                           bounds=([0.0, 1e-6, 0.1, -10.0], [100.0, 100.0, 10.0, 10.0]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    vmax, k, n, c = _params
    x = max(float(row['x_ind']), 1e-12)
    u = float(row['u_ctrl'])
    return float(vmax * x**n / (k**n + x**n) + c * u)