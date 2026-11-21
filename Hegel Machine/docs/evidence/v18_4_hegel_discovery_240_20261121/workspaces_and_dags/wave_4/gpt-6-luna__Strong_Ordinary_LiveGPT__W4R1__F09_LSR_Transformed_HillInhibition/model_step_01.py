import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    hill = (u * u) / (k_half + u * u)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 2.0, 0.5, 0.1], [2.0, 1.0, 1.0, 0.2], [8.0, 5.0, 0.2, 0.05]):
        result = least_squares(residual, start, bounds=([1e-8, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))