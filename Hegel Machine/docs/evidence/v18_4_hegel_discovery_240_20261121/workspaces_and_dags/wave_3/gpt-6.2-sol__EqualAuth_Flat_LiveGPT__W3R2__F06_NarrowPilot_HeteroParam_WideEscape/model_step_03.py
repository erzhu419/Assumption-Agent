import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    base = u / (1.2 + u)
    grad = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        a, b, c = p
        return (a - b * grad) * base + c * root - y

    _params = least_squares(residual, [1.0, 1.0, 0.0]).x

def predict(row) -> float:
    a, b, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float((a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))
