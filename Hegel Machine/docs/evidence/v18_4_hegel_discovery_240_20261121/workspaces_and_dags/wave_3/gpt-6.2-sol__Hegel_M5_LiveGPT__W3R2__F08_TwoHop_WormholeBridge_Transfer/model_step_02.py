import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    root_u = np.sqrt(u)

    def residual(p):
        a, b, k, c, d = p
        return (a + b / (1.0 + k * s)) * u / (c + u) + d * root_u - y

    _params = least_squares(residual, [2.0, 4.0, 1.0, 1.0, 0.4], bounds=([0.0, 0.0, 1e-8, 1e-8, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    a, b, k, c, d = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float((a + b / (1.0 + k * s)) * u / (c + u) + d * math.sqrt(u))