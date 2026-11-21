import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

def _response(u, s, p):
    vmax, half_sat, amplitude, u_mid, s_mid = p
    base = vmax * (1.0 - 0.705 * s / (1.0 + s)) * u / (half_sat + u)
    branch = amplitude * _sigmoid(1.5 * (u - u_mid)) * _sigmoid(5.0 * (s - s_mid))
    return base + branch

def fit(train_rows):
    global _params
    data = [(float(r['u_now']) / 4.0, float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_response(u, s, p) - y for u, s, y in data], dtype=float)
    result = least_squares(residuals, [7.3, 1.2, 0.35, 1.3, 0.75], bounds=([0.0, 0.05, 0.0, 0.0, 0.0], [30.0, 6.0, 2.0, 5.0, 4.0]), x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_response(float(row['u_now']) / 4.0, float(row['s_grad']), _params))