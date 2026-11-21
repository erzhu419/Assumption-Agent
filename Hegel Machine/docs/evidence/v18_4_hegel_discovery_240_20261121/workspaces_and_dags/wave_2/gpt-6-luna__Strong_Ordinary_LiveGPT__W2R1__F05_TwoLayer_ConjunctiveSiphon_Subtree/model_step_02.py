import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.05
_PARAMS = None
_BRANCH = None

def _base(row, p, branch):
    vmax, rho, c = p
    un = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    core = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt_plus':
        return core + c * math.sqrt(max(un, 0.0))
    if branch == 'sqrt_minus':
        return core - c * math.sqrt(max(un, 0.0))
    return core - c * un

def fit(train_rows):
    global _PARAMS, _BRANCH
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    candidates = []
    for branch in ('sqrt_plus', 'sqrt_minus', 'linear_leak'):
        def residual(p):
            return np.asarray([_base(r, p, branch) for r in rows]) - y
        res = least_squares(residual, x0=np.array([5.0, 0.5, 0.1]),
                            bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]),
                            max_nfev=20000)
        candidates.append((float(np.dot(res.fun, res.fun)), branch, res.x))
    _, _BRANCH, _PARAMS = min(candidates, key=lambda item: item[0])
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_base(row, _PARAMS, _BRANCH))