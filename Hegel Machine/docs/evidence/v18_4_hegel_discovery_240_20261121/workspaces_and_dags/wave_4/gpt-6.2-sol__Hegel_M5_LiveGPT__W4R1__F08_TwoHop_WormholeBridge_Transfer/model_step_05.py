import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _transfer(p, u, s):
    a, k, b, c, d, r = p
    return a * u / ((k + u) * (1.0 + b * s)) - d * np.expm1(-r * u) / (1.0 + c * s * u)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(t['u_now'], t['s_grad'], t['y']) for t in train_rows], dtype=float).T
    result = least_squares(lambda p: _transfer(p, u, s) - y, [4.5, 1.0, 1.0, 0.01, 1.0, 1.0], bounds=(1e-9, np.inf), x_scale='jac', max_nfev=1000)
    _params = result.x
    return _params

def predict(row) -> float:
    a, k, b, c, d, r = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(a * u / ((k + u) * (1.0 + b * s)) - d * math.expm1(-r * u) / (1.0 + c * s * u))