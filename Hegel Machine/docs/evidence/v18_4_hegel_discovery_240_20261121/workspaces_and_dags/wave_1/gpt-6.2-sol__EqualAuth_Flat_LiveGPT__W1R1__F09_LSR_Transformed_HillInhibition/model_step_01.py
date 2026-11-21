import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float) / TAU_SCALE
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _law(p, u, s) - y,
                           x0=[2.0 * np.max(y), 1.0, 1.0, 0.1],
                           bounds=([0.0, 1e-9, 0.0, 0.0], np.inf),
                           x_scale='jac')
    _params = result.x

def predict(row):
    return float(_law(_params, row['u_now'] / TAU_SCALE, row['s_grad']))
