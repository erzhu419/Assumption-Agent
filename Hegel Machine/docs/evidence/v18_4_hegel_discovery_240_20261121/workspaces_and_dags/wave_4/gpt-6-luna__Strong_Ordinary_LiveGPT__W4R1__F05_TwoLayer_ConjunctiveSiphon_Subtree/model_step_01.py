import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _base(row):
    un = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return (un / (1.2 + un)), (s / (1.0 + s)), un

def fit(train_rows):
    global _PARAMS, _BRANCH
    ys = np.array([float(r['y']) for r in train_rows])
    candidates = []
    # The branch term may be an open local sqrt contribution, its negative,
    # or linear mass-drift leakage. Fit each structure numerically.
    for branch in ('sqrt', 'negative_sqrt', 'leak'):
        def pred(p, row):
            sat, inhib, un = _base(row)
            vmax, rho, c = p
            b = math.sqrt(max(un, 0.0)) if branch in ('sqrt', 'negative_sqrt') else -un
            if branch == 'negative_sqrt':
                b = -b
            return vmax * (1.0 - rho * inhib) * sat + c * b
        def residual(p):
            return np.array([pred(p, r) - float(r['y']) for r in train_rows])
        result = least_squares(residual, x0=[4.0, 0.5, 0.1],
                               bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]),
                               max_nfev=20000)
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, _BRANCH, _PARAMS = min(candidates, key=lambda item: item[0])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    sat, inhib, un = _base(row)
    vmax, rho, c = _PARAMS
    if _BRANCH == 'sqrt':
        b = math.sqrt(max(un, 0.0))
    elif _BRANCH == 'negative_sqrt':
        b = -math.sqrt(max(un, 0.0))
    else:
        b = -un
    return float(vmax * (1.0 - rho * inhib) * sat + c * b)
