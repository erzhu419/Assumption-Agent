import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_evaluate(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([5.0, 1.0, 1.0, 0.1], [10.0, 3.0, 2.0, 0.5], [3.0, 0.3, 0.3, 0.01]):
        result = least_squares(residual, start, bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_evaluate(row, _params))