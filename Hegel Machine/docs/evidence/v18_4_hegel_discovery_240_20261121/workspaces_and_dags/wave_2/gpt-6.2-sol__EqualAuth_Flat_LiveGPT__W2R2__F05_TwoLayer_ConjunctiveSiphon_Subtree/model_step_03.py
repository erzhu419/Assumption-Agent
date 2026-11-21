import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, p):
    a, b, c, log_k = p
    return (a - b * s / (1.0 + s)) * u / (math.exp(log_k) + u) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _response(u, s, p) - y, [5.0, 3.0, 0.0, math.log(3.0)], bounds=([0.0, 0.0, -np.inf, -20.0], [np.inf, np.inf, np.inf, 20.0]), x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(float(row['u_now']), float(row['s_grad']), _params))
