import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    dy = np.array([r['y'] - r['x_now'] for r in train_rows], dtype=float)
    def residual(p):
        c, a, k = p
        return d * (c + a * np.exp(-k * h)) - dy
    _params = least_squares(residual, [0.08, 0.33, 0.8], bounds=([0, 0, 0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    c, a, k = _params
    d = row['u_now'] - row['u_past']
    return float(row['x_now'] + d * (c + a * math.exp(-k * row['H'])))