import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in train_rows])
    best = None
    for start in ([6, 1, 1, 0.1], [10, 2, 0.5, 0.2], [5, 0.1, 2, 0.01]):
        result = least_squares(residual, start, bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, row))