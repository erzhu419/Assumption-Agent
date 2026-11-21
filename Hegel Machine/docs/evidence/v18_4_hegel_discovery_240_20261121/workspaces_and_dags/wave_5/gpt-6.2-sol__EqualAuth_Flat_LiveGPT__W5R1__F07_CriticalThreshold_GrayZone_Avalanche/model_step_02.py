import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    a, b, c, k, threshold = p
    return (a - b * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u) - k * u * np.maximum(0.0, s - threshold)

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(u, s, p) - y, [5., 3., 0.5, 1., 2.6], bounds=([0., 0., 0., 0., 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]), x_scale='jac').x

def predict(row) -> float:
    u, s = float(row['u_now']), float(row['s_grad'])
    a, b, c, k, threshold = _params
    return float((a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u) - k * u * max(0.0, s - threshold))
