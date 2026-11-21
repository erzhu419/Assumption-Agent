import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-12, float(row['u_now']))
    s = float(row['s_grad'])
    base = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return base + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = ([3.0, 1.0, 0.5, 0.1], [5.0, 2.0, 0.7, -0.2], [2.0, 0.2, 0.2, 0.5])
    best = None
    for x0 in starts:
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            x0=np.asarray(x0, dtype=float),
            bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 2.0, np.inf]),
            max_nfev=20000
        )
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))