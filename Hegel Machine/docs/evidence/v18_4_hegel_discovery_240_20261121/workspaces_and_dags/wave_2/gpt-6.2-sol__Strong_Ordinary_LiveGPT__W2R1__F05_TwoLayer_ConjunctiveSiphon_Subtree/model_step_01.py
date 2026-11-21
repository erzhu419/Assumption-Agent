import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.05 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    X = np.column_stack((base, -base * s / (1.0 + s), np.sqrt(u)))
    _params = least_squares(lambda p: X @ p - y, [7.0, 5.0, 0.5], jac=lambda p: X).x

def predict(row) -> float:
    v, v_rho, c = _params
    u = row['u_now'] / 0.05
    s = row['s_grad']
    return float((v - v_rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))