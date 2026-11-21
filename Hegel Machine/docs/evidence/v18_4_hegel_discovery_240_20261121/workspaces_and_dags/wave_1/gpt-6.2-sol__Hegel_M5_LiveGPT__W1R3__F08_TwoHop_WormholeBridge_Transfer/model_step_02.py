import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float)
    u, s, y = data.T
    saturation = u / (1.2 + u)
    gradient = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, coeff = p
        return vmax * (1.0 - rho * gradient) * saturation + coeff * root - y

    _params = least_squares(residual, [6.5, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x
    return _params

def predict(row):
    vmax, rho, coeff = _params
    u, s = row['u_now'], row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + coeff * math.sqrt(u))
