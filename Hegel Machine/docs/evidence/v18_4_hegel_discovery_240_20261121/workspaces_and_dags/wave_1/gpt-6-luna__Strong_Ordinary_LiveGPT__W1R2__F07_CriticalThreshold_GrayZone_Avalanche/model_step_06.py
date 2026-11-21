import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - ys
    starts = [
        [5.0, 1.0, 0.5, 0.5],
        [10.0, 3.0, 1.0, 1.0],
        [3.0, 0.2, 2.0, 0.8],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda z: float(np.sum(z.fun ** 2))).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))