import math
import numpy as np
from scipy.optimize import least_squares

_tau_scale = 3.0
_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _tau_scale
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    starts = ([3.0, 1.0, 0.70, 0.2], [1.0, 0.2, 0.70, 0.8], [5.0, 3.0, 0.70, 0.05])
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            x0=np.asarray(x0), bounds=([0.0, 0.0, 0.05, 0.0], [np.inf, np.inf, 0.95, np.inf]),
                            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))