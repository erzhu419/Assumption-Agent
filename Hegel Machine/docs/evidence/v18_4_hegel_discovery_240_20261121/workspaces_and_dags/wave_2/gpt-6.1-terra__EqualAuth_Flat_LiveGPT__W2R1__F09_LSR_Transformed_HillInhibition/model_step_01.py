import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    vmax, k_half, alpha, c_leak = p
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - c_leak * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in train_rows])
    best = None
    for k in (0.1, 1.0, 10.0):
        for alpha in (0.1, 1.0, 3.0):
            result = least_squares(residual, [10.0, k, alpha, 0.1], bounds=(0.0, np.inf), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, row))