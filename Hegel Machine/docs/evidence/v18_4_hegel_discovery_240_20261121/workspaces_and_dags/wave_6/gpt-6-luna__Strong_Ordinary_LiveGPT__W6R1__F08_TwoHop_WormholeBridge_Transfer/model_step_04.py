import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _base(row, vmax, rho):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))

def fit(train_rows):
    global _PARAMS, _BRANCH
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    candidates = []
    for branch in ('sqrt_plus', 'sqrt_minus', 'leak'):
        def pred(p, r):
            vmax, rho, c = p
            un = float(r['u_now'])
            val = _base(r, vmax, rho)
            if branch == 'sqrt_plus':
                val += c * math.sqrt(max(un, 0.0))
            elif branch == 'sqrt_minus':
                val -= c * math.sqrt(max(un, 0.0))
            else:
                val -= c * un
            return val
        def residual(p):
            return np.array([pred(p, r) for r in rows]) - y
        res = least_squares(residual, x0=[5.0, 0.5, 0.2], bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]), max_nfev=20000)
        candidates.append((float(np.dot(res.fun, res.fun)), branch, res.x))
    _, _BRANCH, _PARAMS = min(candidates, key=lambda item: item[0])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c = [float(v) for v in _PARAMS]
    un = float(row['u_now'])
    val = _base(row, vmax, rho)
    if _BRANCH == 'sqrt_plus':
        val += c * math.sqrt(max(un, 0.0))
    elif _BRANCH == 'sqrt_minus':
        val -= c * math.sqrt(max(un, 0.0))
    else:
        val -= c * un
    return float(val)