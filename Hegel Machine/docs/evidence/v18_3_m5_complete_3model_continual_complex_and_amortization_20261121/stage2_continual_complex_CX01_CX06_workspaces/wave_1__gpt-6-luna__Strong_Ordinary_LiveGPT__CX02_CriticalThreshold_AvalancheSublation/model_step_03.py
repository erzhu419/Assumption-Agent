import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _value(row, p, branch):
    vmax, rho, branch_amp = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt':
        return base - branch_amp * math.sqrt(max(u, 0.0))
    return base - branch_amp * u

def fit(train_rows):
    global _PARAMS, _BRANCH
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    best = None
    for branch in ('sqrt', 'linear_leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in rows]) - y
        for initial in ([4.0, 1.0, 0.5], [6.0, 1.8, 1.0], [3.0, 2.5, 0.1]):
            result = least_squares(residual, np.asarray(initial), max_nfev=20000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, result.x.copy(), branch)
    _PARAMS, _BRANCH = best[1], best[2]
    return {'params': _PARAMS, 'branch': _BRANCH, 'sse': best[0]}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS, _BRANCH))