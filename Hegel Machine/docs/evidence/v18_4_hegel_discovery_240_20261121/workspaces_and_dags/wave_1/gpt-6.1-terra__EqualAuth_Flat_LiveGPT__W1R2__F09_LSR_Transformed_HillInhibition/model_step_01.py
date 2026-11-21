import numpy as np
from scipy.optimize import least_squares

params = None

def fit(train_rows):
    global params
    u = np.array([r['u_now'] / 2.5 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u - y
    best = None
    for start in ([5, 1, 1, 0.1], [10, 2, 0.5, 0.5], [3, 0.2, 2, 0.01], [20, 5, 2, 1]):
        result = least_squares(residual, start, bounds=(0, np.inf), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    params = best.x
    return params

def predict(row):
    vmax, k_half, alpha, leak = params
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u)
