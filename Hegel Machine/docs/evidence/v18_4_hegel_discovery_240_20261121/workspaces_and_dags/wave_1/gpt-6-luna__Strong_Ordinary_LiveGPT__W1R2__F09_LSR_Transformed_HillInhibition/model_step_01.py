import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = ([5, 1, 0.3, 0.1], [10, 5, 1, 0.01], [3, 0.2, 0.05, 0.5])
    candidates = [least_squares(residual, x0, bounds=([1e-8]*4, [1e4]*4), max_nfev=20000) for x0 in starts]
    _params = min(candidates, key=lambda result: np.sum(result.fun**2)).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))