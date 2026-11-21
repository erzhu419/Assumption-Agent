import numpy as np
from scipy.optimize import least_squares

def _model(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_model(p, r) for r in train_rows]) - y
    starts = [np.array([4.0, 1.0, 0.5, 0.1]), np.array([3.0, 2.0, 0.2, 0.0]), np.array([6.0, 0.5, 1.0, 0.2])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return float(_model(fit(_TRAIN_ROWS), row))