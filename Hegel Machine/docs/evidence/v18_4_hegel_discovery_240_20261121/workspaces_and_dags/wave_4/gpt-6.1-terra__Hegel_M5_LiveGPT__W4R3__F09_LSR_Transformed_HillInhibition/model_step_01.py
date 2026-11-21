import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    results = [least_squares(residual, start, bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000) for start in ([6, 1, 0.5, 0.1], [10, 2, 1, 0.3], [5, 0.1, 0.2, 0.01])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_model(row, _params))