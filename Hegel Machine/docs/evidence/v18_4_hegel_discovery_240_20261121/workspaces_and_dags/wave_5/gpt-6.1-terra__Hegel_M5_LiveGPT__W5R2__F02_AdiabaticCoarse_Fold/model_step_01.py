import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        equilibrium = p[0] * u + p[1]
        return equilibrium + (x - equilibrium) * np.exp(-p[2] * h) - y
    result = least_squares(residual, [2.1, 0.0, 1.2], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x

def predict(row):
    a, b, rate = _params
    equilibrium = a * row['u_now'] + b
    return float(equilibrium + (row['x_now'] - equilibrium) * math.exp(-rate * row['H']))