import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def _law(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float) / TAU_SCALE
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _law(u, s, p) - y, [6.0, 1.0, 0.5, 0.1], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_law(float(row['u_now']) / TAU_SCALE, float(row['s_grad']), _params))
