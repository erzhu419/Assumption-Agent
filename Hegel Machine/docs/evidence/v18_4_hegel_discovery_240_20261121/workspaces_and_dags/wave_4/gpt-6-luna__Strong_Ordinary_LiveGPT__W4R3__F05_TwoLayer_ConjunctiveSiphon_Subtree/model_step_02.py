import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _base(row, vmax, rho):
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))

def fit(train_rows):
    global _PARAMS, _BRANCH
    y = np.asarray([float(r['y']) for r in train_rows])
    candidates = []
    # The source-hub law has either a positive square-root local branch or linear leak.
    for branch in ('sqrt', 'leak'):
        def pred(p, row):
            u = float(row['u_now']) / 0.05
            extra = p[2] * math.sqrt(u) if branch == 'sqrt' else -p[2] * u
            return _base(row, p[0], p[1]) + extra
        def residual(p):
            return np.asarray([pred(p, r) for r in train_rows]) - y
        result = least_squares(residual, x0=[5.0, 0.5, 0.2], bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, _BRANCH, _PARAMS = min(candidates, key=lambda item: item[0])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, rho, branch_amp = _PARAMS
    u = float(row['u_now']) / 0.05
    extra = branch_amp * math.sqrt(u) if _BRANCH == 'sqrt' else -branch_amp * u
    return float(_base(row, vmax, rho) + extra)
