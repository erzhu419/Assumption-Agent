import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    u, s = _features(row)
    j0, beta_bar, rho, c_sqrt = p
    un = u  # characteristic_tau_scale = 1.0
    return j0 * un * math.exp(-beta_bar / (0.5 + un)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(un)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = ([3.0, 1.0, 0.5, 0.2], [5.0, 2.0, 0.7, 0.0], [1.0, 0.2, 0.2, 0.5])
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            x0=np.asarray(x0, dtype=float),
                            bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 20.0, 1.0, 10.0]),
                            max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))