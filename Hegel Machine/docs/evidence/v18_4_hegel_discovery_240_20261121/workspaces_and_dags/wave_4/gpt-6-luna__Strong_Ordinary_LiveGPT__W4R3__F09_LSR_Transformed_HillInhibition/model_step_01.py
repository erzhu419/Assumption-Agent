import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _basis(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    starts = [np.array([6.0, 1.0, 0.1, 0.05]), np.array([10.0, 4.0, 0.5, 0.01]), np.array([5.0, 10.0, 1.0, 0.1])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.array([_basis(r, p) for r in train_rows]) - ys,
                            x0, bounds=([1e-8, 1e-8, 0.0, 0.0], [1e4, 1e4, 1e3, 1e3]),
                            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_basis(row, _params))