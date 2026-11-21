import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        c, a, log_k = p
        k = math.exp(log_k)
        return np.array([r['x_now'] + (r['u_now'] - r['u_past']) * (c + a * math.exp(-k * r['H'])) - r['y'] for r in train_rows])
    result = least_squares(residual, [0.6, 2.5, math.log(1.5)])
    _params = result.x

def predict(row):
    c, a, log_k = _params
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * (c + a * math.exp(-math.exp(log_k) * row['H'])))