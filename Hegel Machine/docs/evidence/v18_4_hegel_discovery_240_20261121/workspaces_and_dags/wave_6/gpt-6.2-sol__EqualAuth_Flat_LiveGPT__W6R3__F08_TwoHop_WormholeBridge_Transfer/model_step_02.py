import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    def residual(p):
        direct, gated, half_sat, loss = p
        return (direct + gated / (1.0 + s)) * u / (half_sat + u) - loss * root_u - y
    _params = least_squares(residual, [2.0, 5.0, 1.0, 0.4], bounds=([0.0, 0.0, 1e-8, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    direct, gated, half_sat, loss = _params
    u, s = row['u_now'], row['s_grad']
    return float((direct + gated / (1.0 + s)) * u / (half_sat + u) - loss * math.sqrt(u))