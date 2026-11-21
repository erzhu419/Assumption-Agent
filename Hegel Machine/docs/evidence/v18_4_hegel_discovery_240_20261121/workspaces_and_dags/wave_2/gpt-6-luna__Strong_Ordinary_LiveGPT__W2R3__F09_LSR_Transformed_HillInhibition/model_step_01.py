import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    starts = [np.array([4.0, 1.0, 0.5, 0.05]), np.array([8.0, 4.0, 0.2, 0.1]), np.array([5.0, 0.2, 1.0, 0.01])]
    best = None
    for p0 in starts:
        result = least_squares(lambda p: np.array([_pred(r, p) for r in train_rows]) - y,
                               p0, bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
                               max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return float(_pred(row, _FIT_PARAMS))