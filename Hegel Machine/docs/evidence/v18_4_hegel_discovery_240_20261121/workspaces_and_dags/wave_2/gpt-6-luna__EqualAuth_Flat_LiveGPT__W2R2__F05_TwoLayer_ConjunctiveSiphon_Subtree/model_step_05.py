import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _features(row):
    un = max(float(row['u_now']) / 4.0, 1e-8)
    s = max(float(row['s_grad']), 0.0)
    base = (1.0 - 0.70 * s / (1.0 + s)) * (un / (1.2 + un))
    return un, base

def _value(row, p, branch):
    un, base = _features(row)
    vmax, rho, q = p
    sink = (1.0 - rho * float(row['s_grad']) / (1.0 + float(row['s_grad'])))
    core = vmax * sink * (un / (1.2 + un))
    if branch == 'sqrt':
        return core + q * math.sqrt(un)
    return core - q * un

def fit(train_rows):
    global _params, _branch
    y = np.asarray([float(r['y']) for r in train_rows])
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in train_rows]) - y
        starts = ([3.0, 0.5, 0.2], [4.0, 0.8, 0.5], [2.0, 0.2, 0.0])
        for x0 in starts:
            res = least_squares(residual, x0=x0, bounds=([0.0, 0.0, -10.0 if branch == 'sqrt' else 0.0], [100.0, 2.0, 10.0]))
            score = float(np.sum(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, branch, res.x)
    _, _branch, _params = best
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params, _branch))