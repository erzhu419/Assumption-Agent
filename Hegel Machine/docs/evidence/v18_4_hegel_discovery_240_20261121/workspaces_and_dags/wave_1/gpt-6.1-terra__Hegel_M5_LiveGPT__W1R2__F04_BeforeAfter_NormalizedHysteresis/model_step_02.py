import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        a, b, k = p
        return np.array([r['x_now'] + (r['u_now'] - r['u_past']) * (a + b * math.exp(-k * r['H'])) - r['y'] for r in train_rows])
    result = least_squares(residual, [0.65, 1.75, 0.8], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x

def predict(row):
    a, b, k = _params
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * (a + b * math.exp(-k * row['H'])))