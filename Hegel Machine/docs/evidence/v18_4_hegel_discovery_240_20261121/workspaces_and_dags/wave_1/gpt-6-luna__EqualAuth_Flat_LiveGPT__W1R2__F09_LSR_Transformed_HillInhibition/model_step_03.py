import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 2.5

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = ([5.0, 1.0, 0.5, 0.1], [8.0, 2.0, 1.0, 0.01], [3.0, 0.2, 0.1, 0.3])
    fits = [least_squares(residual, x0, bounds=([0, 1e-10, 0, 0], [100, 100, 100, 100]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: np.sum(result.fun ** 2))
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))