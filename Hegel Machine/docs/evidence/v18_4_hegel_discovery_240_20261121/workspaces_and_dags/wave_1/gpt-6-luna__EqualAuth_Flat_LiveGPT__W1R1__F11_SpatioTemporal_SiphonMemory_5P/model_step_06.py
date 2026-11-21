import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = 1.0

def _value(row, p, branch):
    vmax, rho, c = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + branch * c * math.sqrt(u)

def fit(train_rows):
    global _PARAMS, _BRANCH
    best = None
    for branch in (1.0, -1.0):
        def residual(p):
            return np.array([_value(r, p, branch) - float(r['y']) for r in train_rows])
        result = least_squares(residual, x0=[2.0, 0.7, 0.2], bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 20.0]))
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x.copy(), branch)
    _PARAMS, _BRANCH = best[1], best[2]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_value(row, _PARAMS, _BRANCH))