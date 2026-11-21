import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        vmax, rho, c = p
        return np.array([vmax * (1 - rho * r['s_grad'] / (1 + r['s_grad'])) * ((r['u_now'] / 0.05) / (1.2 + r['u_now'] / 0.05)) + c * math.sqrt(r['u_now'] / 0.05) - r['y'] for r in train_rows])
    result = least_squares(residual, [6.5, 0.7, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x

def predict(row):
    vmax, rho, c = _params
    u = row['u_now'] / 0.05
    s = row['s_grad']
    return float(vmax * (1 - rho * s / (1 + s)) * u / (1.2 + u) + c * math.sqrt(u))