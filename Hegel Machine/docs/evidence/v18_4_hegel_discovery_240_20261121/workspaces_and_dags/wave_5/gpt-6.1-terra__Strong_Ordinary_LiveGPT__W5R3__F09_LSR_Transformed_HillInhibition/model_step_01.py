import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, k_half, alpha, leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    results = [least_squares(residual, start, bounds=(0, np.inf), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12) for start in ([8, 1, 1, 0.1], [5, 0.2, 0.5, 0.01], [15, 4, 2, 0.5])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_model(_params, row))