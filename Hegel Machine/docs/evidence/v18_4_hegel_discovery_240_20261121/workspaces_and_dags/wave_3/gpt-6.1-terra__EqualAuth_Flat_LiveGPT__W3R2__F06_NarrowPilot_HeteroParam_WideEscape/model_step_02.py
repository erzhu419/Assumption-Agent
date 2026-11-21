import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        a, b, c = p
        return np.array([a * r['u_now'] / (1.2 + r['u_now']) - b * r['s_grad'] / (1.0 + r['s_grad']) * r['u_now'] / (1.2 + r['u_now']) + c * math.sqrt(max(0.0, r['u_now'])) - r['y'] for r in train_rows])
    _params = least_squares(residual, [5.0, 3.0, 0.5]).x

def predict(row):
    a, b, c = _params
    u = row['u_now']
    s = row['s_grad']
    return float((a - b * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(0.0, u)))
