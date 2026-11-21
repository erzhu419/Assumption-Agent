import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    drive = u / (1.2 + u)
    return (vmax * sink * drive * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    lower = np.array([0.01, 0.0, 1e-4, 0.0, 1e-4])
    upper = np.array([30.0, 0.999, 100.0, 10.0, 500.0])
    starts = [
        np.array([6.0, 0.7, 1.0, 0.5, 10.0]),
        np.array([5.0, 0.5, 0.5, 0.3, 20.0]),
        np.array([8.0, 0.8, 2.0, 1.0, 30.0]),
    ]
    best = None
    for x0 in starts:
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            np.clip(x0, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=20000, x_scale='jac')
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))