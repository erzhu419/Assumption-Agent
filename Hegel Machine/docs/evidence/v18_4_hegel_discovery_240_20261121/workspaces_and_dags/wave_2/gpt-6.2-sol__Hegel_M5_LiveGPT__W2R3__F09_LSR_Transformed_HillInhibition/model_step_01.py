import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.1
_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float) / TAU
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    q = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * q / ((k_half + q) * (1.0 + alpha * s)) - leak * u - y

    _params = least_squares(residual, [6.0, 1.0, 1.0, 0.2], bounds=(0.0, np.inf), x_scale='jac').x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / TAU
    q = math.pow(u, 2)
    return float(vmax * q / ((k_half + q) * (1.0 + alpha * float(row['s_grad']))) - leak * u)
