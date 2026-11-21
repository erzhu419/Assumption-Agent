import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def model(p, uu, ss):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * (uu * uu / (k_half + uu * uu)) / (1.0 + alpha_inh * ss) - c_leak * uu
    scale = max(1.0, float(np.max(np.abs(y))))
    starts = [[max(scale, 1.0), 1.0, 0.5, 0.05], [2*scale, 4.0, 1.0, 0.2], [5*scale, 10.0, 0.1, 0.0]]
    best = None
    for p0 in starts:
        res = least_squares(lambda p: model(p, u, s) - y, p0,
                            bounds=([1e-9, 1e-9, 0.0, 0.0], [1e4, 1e4, 1e3, 1e3]),
                            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict requires fitted parameters; call fit(train_rows) and use its result')
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, k_half, alpha_inh, c_leak = p
    return float(vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u)
