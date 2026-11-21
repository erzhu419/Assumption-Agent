import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    H = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([_pred(r, p) for r in rows]) - y
    best = None
    # Multiple positive-scale initializations reduce the risk of a local fit
    # under the symmetric exchange of fast and slow components.
    for start in ([1.0, 0.5, 1.0, 10.0], [1.0, 1.0, 1.0, 20.0], [2.0, 0.2, 1.0, 8.0]):
        result = least_squares(residual, np.array(start, dtype=float),
            bounds=([0.0, 1e-4, 0.0, 1e-4], [20.0, 1e3, 20.0, 1e4]),
            max_nfev=20000, x_scale='jac')
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_pred(row, _PARAMS))
