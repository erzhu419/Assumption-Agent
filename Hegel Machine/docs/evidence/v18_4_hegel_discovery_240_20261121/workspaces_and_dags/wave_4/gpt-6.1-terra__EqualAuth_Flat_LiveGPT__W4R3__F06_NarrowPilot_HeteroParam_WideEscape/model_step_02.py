import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([p[0] * r['u_now'] / (1.2 + r['u_now']) - p[1] * r['s_grad'] / (1.0 + r['s_grad']) * r['u_now'] / (1.2 + r['u_now']) + p[2] * np.sqrt(max(0.0, r['u_now'])) - r['y'] for r in train_rows])
    _params = least_squares(residual, [6.0, 1.0, 1.0]).x

def predict(row):
    u = row['u_now']
    s = row['s_grad']
    return float((_params[0] - _params[1] * s / (1.0 + s)) * u / (1.2 + u) + _params[2] * np.sqrt(max(0.0, u)))
