import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _value(row, p, branch):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt':
        return base + c * math.sqrt(max(un, 0.0))
    return base - c * un

def fit(train_rows):
    global _PARAMS, _BRANCH
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, x0=np.array([6.0, 0.5, 0.3]), bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 20.0]))
        loss = float(np.mean(residual(result.x) ** 2))
        if best is None or loss < best[0]:
            best = (loss, branch, result.x)
    _PARAMS = best[2]
    _BRANCH = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS, _BRANCH))