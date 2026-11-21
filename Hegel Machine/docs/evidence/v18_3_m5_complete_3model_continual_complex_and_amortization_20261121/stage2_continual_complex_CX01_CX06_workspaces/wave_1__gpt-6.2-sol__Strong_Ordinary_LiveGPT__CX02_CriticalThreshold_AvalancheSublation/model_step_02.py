import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, p):
    vmax, rho, local, loss, threshold = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + local * np.sqrt(u) - loss * u * np.maximum(s - threshold, 0.0)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(u, s, p) - y, [7.0, 0.7, 0.3, 1.0, 1.4], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    u, s = float(row['u_now']), float(row['s_grad'])
    vmax, rho, local, loss, threshold = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + local * math.sqrt(u) - loss * u * max(s - threshold, 0.0))
