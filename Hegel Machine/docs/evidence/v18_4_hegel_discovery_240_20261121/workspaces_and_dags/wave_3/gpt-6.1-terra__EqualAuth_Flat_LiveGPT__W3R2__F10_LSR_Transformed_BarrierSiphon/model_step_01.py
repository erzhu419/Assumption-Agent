import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, rows):
    j0, beta, rho, c = p
    u = np.array([r['u_now'] for r in rows], dtype=float) / 3.0
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        res = least_squares(lambda p: _model(p, train_rows) - y,
                            [3.0, beta, 0.5, 0.2],
                            bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]),
                            max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x

def predict(row):
    return float(_model(_params, [row])[0])