import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        vmax, rho, c = p
        return np.array([vmax * (1.0 - rho * r['s_grad'] / (1.0 + r['s_grad'])) * r['u_now'] / (1.2 + r['u_now']) + c * math.sqrt(r['u_now']) - r['y'] for r in train_rows])
    _params = least_squares(residual, [8.0, 0.8, 0.3]).x

def predict(row):
    vmax, rho, c = _params
    u = row['u_now']
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))