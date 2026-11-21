import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow, rho = p
    h = float(row['H'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    sink = 1.0 - rho * s / (1.0 + s)
    temporal = amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow))
    return u * sink * temporal

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = [
        [1.0, 2.0, 1.0, 20.0, 0.3],
        [0.8, 1.0, 1.2, 15.0, 0.5],
        [1.5, 4.0, 1.5, 30.0, 0.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0,
            bounds=([0.0, 1e-4, 0.0, 1e-4, -2.0], [20.0, 100.0, 20.0, 200.0, 2.0]),
            max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _PARAMS))