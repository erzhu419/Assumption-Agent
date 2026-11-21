import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = ([2.0, 1.0, 0.70, 0.3], [4.0, 2.0, 0.5, 0.1], [1.0, 0.2, 0.9, 0.5])
    best = None
    for x0 in starts:
        res = least_squares(
            lambda p: np.asarray([_value(r, p) for r in train_rows]) - y,
            x0=np.asarray(x0, dtype=float),
            bounds=([0.0, 0.0, 0.05, -10.0], [100.0, 50.0, 0.95, 10.0]),
            max_nfev=20000
        )
        loss = float(np.dot(res.fun, res.fun))
        if best is None or loss < best[0]:
            best = (loss, res.x)
    _params = best[1]
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))