import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return base - adapt

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [np.array([6.0, 0.70, 0.30, 0.5, 10.0]), np.array([5.0, 0.7, 0.2, 1.0, 20.0]), np.array([8.0, 0.7, 0.5, 0.3, 8.0])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.01, 0.0, 0.0, 0.01, 0.1], [30.0, 0.99, 10.0, 10.0, 200.0]), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))