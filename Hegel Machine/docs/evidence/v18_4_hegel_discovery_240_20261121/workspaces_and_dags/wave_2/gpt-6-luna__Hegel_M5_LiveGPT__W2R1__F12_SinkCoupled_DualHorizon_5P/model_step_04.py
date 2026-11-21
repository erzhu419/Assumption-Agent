import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    return vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows], dtype=float) - ys
    starts = [
        [6.0, 0.70, 1.0, 0.8, 12.0],
        [5.0, 0.70, 0.5, 0.5, 20.0],
        [8.0, 0.60, 2.0, 1.0, 30.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0=np.asarray(x0), bounds=([0.01, 0.0, 0.01, 0.0, 0.1], [30.0, 0.999, 20.0, 10.0, 200.0]), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))