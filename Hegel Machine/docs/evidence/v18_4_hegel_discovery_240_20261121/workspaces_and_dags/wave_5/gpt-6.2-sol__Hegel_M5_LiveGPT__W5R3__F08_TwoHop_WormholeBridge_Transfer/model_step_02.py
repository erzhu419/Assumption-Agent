import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    root = np.sqrt(u)
    def residual(p):
        a, b, k, c = p
        return (a + b / (1.0 + s)) * u / (k + u) + c * root - y
    _params = least_squares(residual, [2.0, 5.0, 1.0, 0.3], bounds=([0.0, 0.0, 1e-8, 0.0], np.inf)).x

def predict(row) -> float:
    a, b, k, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float((a + b / (1.0 + s)) * u / (k + u) + c * math.sqrt(u))