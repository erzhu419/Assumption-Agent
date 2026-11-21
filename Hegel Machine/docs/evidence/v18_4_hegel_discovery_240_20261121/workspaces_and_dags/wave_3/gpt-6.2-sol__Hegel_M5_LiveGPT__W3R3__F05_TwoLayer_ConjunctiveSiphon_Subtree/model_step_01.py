import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _terms(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    a = u / (1.2 + u)
    return a, -a * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([_terms(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(p):
        v, rho, c = p
        return X @ np.array([v, v * rho, c]) - y

    _params = least_squares(residual, [6.5, 0.70, 0.14], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x
    return _params

def predict(row):
    a, b, root_u = _terms(row)
    v, rho, c = _params
    return float(v * (a + rho * b) + c * root_u)
