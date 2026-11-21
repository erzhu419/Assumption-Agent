import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)

    def residual(p):
        floor, gain, k, attenuation, bypass = p
        return (floor + gain / (1.0 + attenuation * s)) * u / (k + u) + bypass * root_u - y

    _params = least_squares(residual, [2.0, 5.0, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 1e-8, 0.0, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    floor, gain, k, attenuation, bypass = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float((floor + gain / (1.0 + attenuation * s)) * u / (k + u) + bypass * math.sqrt(u))