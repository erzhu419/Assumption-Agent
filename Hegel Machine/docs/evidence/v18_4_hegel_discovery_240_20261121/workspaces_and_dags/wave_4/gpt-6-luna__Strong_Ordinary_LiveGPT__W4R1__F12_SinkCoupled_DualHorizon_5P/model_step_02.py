import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = [
        [3.0, 0.7, 0.5, 1.0, 12.0],
        [2.0, 0.4, 0.3, 0.5, 20.0],
        [4.0, 0.9, 0.8, 2.0, 30.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            x0=x0,
            bounds=([0.0, 0.0, 0.0, 0.01, 0.1], [100.0, 1.0, 20.0, 20.0, 200.0]),
            max_nfev=20000,
        )
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _PARAMS = best.x
    return {'vmax': float(_PARAMS[0]), 'rho': float(_PARAMS[1]), 'amp_adapt': float(_PARAMS[2]), 'tau_fast': float(_PARAMS[3]), 'tau_slow': float(_PARAMS[4])}

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))