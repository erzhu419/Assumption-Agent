import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * u * u / (k_half + u * u) / (1.0 + alpha * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in rows])
    best = None
    for k in (0.2, 1.0, 4.0):
        for alpha in (0.1, 1.0, 3.0):
            result = least_squares(residual, [5.0, k, alpha, 0.1], bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(row, _params))