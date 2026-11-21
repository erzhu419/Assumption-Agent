import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / 0.1 for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p, uu, ss):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (uu ** 2 / (k_half + uu ** 2)) / (1.0 + alpha_inh * ss) - c_leak * uu

    def residual(p):
        return predict_params(p, u, s) - y

    best = None
    for start in ([4.0, 2.0, 0.5, 0.01], [5.0, 5.0, 1.0, 0.1], [3.0, 0.5, 0.2, 0.001]):
        result = least_squares(residual, start, bounds=([1e-9, 1e-9, 0.0, 0.0], ([np.inf] * 4)))
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    vmax, k_half, alpha_inh, c_leak = p
    return float(vmax * (u ** 2 / (k_half + u ** 2)) / (1.0 + alpha_inh * s) - c_leak * u)