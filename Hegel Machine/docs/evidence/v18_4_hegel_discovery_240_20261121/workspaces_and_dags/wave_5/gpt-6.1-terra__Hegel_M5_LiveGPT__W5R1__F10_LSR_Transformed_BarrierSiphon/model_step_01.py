import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    j0, beta, rho, c = p
    u = np.array([r['u_now'] / 0.02 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(lambda p: _evaluate(p, rows) - y, [3.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(_params, [row])[0])