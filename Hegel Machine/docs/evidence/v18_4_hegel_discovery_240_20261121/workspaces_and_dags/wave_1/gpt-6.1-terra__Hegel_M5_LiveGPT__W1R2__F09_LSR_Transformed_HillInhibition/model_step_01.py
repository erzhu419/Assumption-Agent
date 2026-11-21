import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] / 2.5 for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u - y

    best = None
    for k in (0.3, 1.0, 3.0):
        for alpha in (0.3, 1.0, 3.0):
            result = least_squares(residual, [6.5, k, alpha, 0.2], bounds=([0.0, 1e-9, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=5000)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x

def predict(row):
    vmax, k_half, alpha_inh, c_leak = _params
    u = row['u_now'] / 2.5
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * row['s_grad']) - c_leak * u)
