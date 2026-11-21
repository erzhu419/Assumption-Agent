import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    f = u / (1.2 + u)
    X = np.column_stack((f, -f * s / (1.0 + s), -np.sqrt(u)))
    _params = least_squares(lambda p: X @ p - y, [7.0, 1.0, 0.3], jac=lambda p: X).x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, sink_strength, sqrt_loss = _params
    return float((vmax - sink_strength * s / (1.0 + s)) * u / (1.2 + u) - sqrt_loss * math.sqrt(u))