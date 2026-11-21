import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    fast = 1.0 - math.exp(-H / tau_fast)
    slow = 1.0 - math.exp(-H / tau_slow)
    return vmax * sink * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    starts = [
        [6.0, 0.70, 0.5, 0.8, 12.0],
        [6.0, 0.70, 1.0, 0.8, 20.0],
        [5.0, 0.65, 0.3, 0.5, 8.0],
    ]
    lower = [0.01, 0.0, 1e-4, 0.0, 1e-3]
    upper = [30.0, 1.0, 20.0, 10.0, 200.0]
    results = [least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000) for x0 in starts]
    _params = min(results, key=lambda result: float(np.sum(result.fun ** 2))).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))