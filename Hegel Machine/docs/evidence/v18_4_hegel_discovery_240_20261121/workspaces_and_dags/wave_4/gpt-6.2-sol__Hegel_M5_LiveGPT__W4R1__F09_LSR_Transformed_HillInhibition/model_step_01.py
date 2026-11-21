import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def _law(u2, u, s, p):
    vmax, k_half, alpha, leak = p
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float) / TAU_SCALE
    u2 = u * u
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _law(u2, u, s, p) - y
    initial = [max(1.0, float(np.max(y))), 1.0, 1.0, 0.1]
    result = least_squares(residual, initial, bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    u = float(row['u_now']) / TAU_SCALE
    return float(_law(math.pow(u, 2), u, float(row['s_grad']), _params))