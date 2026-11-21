import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = [np.array([5.0, 1.0, 0.2, 0.1]), np.array([3.0, 0.5, 0.5, 0.05]), np.array([8.0, 2.0, 0.05, 0.2])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_pred(r, p) for r in train_rows]) - y,
                            x0, bounds=([1e-8, 1e-8, 0.0, 0.0], [np.inf]*4),
                            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(row, p))