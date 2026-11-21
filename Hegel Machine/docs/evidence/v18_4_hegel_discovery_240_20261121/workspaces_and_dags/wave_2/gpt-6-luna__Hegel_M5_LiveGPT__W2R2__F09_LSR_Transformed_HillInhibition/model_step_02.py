import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, params):
    vmax, k_half, alpha_inh, c_leak = params
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _params))