import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p, ui, si):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (ui**2 / (k_half + ui**2)) / (1.0 + alpha_inh * si) - c_leak * ui
    result = least_squares(lambda p: pred(p, u, s) - y,
        x0=np.array([4.0, 1.0, 0.2, 0.1]),
        bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
        max_nfev=20000)
    return tuple(float(v) for v in result.x)

def predict(row):
    vmax, k_half, alpha_inh, c_leak = fit_params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u)