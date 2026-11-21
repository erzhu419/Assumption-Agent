import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    sat = u / (1.2 + u)
    X = np.column_stack((sat, -sat * s / (1.0 + s), np.sqrt(u)))
    global _params
    _params = least_squares(lambda p: X @ p - y, [8.0, 1.0, 0.0], jac=lambda p: X).x
    return _params

def predict(row) -> float:
    a, b, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float((a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))