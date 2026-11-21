import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, k_half, alpha, leak = p
    z = (u / 0.1) ** 2
    return vmax * z / (k_half + z) / (1.0 + alpha * s) - leak * (u / 0.1)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for k in (0.1, 1.0, 10.0):
        for alpha in (0.1, 1.0, 5.0):
            res = least_squares(lambda p: _values(p, u, s) - y,
                                [max(1.0, 2.0 * np.max(y)), k, alpha, 0.05],
                                bounds=([0.0, 1e-9, 0.0, 0.0],
                                        [np.inf, np.inf, np.inf, np.inf]),
                                max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
                best = res
    _params = best.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))
