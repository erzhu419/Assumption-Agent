import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    return float(row['u_now']) * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    starts = [
        [1.0, 0.3, 1.0, 10.0],
        [1.0, 1.0, 1.0, 20.0],
        [2.0, 0.2, 1.0, 8.0],
        [1.0, 0.5, 2.0, 15.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(
            lambda p: np.array([_value(r, p) for r in rows]) - y,
            x0=np.array(x0, dtype=float),
            bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 1e3, 20.0, 1e3]),
            max_nfev=20000,
        )
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    _params = np.array(best[1], dtype=float)
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _value(row, _params)
