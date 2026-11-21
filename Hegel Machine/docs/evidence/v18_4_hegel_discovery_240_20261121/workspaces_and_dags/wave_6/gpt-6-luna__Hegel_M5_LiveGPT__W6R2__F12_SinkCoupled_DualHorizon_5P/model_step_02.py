import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    s = float(row['s_grad'])
    u = float(row['u_now'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [6.0, 0.70, 0.5, 0.8, 10.0],
        [6.0, 0.70, 1.0, 0.8, 20.0],
        [6.0, 0.70, 0.2, 0.8, 5.0],
        [6.0, 0.70, 2.0, 0.8, 30.0],
    ]
    lower = [0.01, 0.0, 0.01, 0.0, 0.02]
    upper = [30.0, 0.999, 20.0, 10.0, 200.0]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = np.asarray(best[1], dtype=float)
    return _params.tolist()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))