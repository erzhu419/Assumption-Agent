import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha, leak = p
    u = float(row['u_now']) / 2.5
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * float(row['s_grad'])) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    best = None
    for k in (0.1, 1.0, 10.0):
        for alpha in (0.1, 1.0, 3.0):
            result = least_squares(residual, [10.0, k, alpha, 0.1], bounds=(0.0, np.inf), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))