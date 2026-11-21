import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, km, c0, amp, s0, power = p
    u = max(float(row['u_now']), 1e-9)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
    branch = (c0 - amp * max(0.0, s - s0) ** power) * math.sqrt(u)
    return base + branch

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    starts = [
        [5.0, 0.7, 1.2, 1.0, 2.0, 1.4, 1.0],
        [5.0, 0.7, 1.2, 1.0, 1.0, 1.2, 2.0],
        [4.0, 0.5, 1.0, 0.5, 2.0, 1.5, 1.5]
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0,
            bounds=([0.0, 0.0, 0.05, -10.0, 0.0, 0.0, 0.25],
                    [20.0, 3.0, 10.0, 10.0, 20.0, 4.0, 4.0]),
            max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))