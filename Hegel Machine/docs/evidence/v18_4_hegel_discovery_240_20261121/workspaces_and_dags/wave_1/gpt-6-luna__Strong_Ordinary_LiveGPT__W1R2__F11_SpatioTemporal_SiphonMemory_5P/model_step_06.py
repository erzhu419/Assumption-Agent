import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def fit(train_rows):
    global _PARAMS, _BRANCH
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, branch):
        vmax, rho, q = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        if branch == 'sqrt_plus':
            return base + q * np.sqrt(u)
        if branch == 'sqrt_minus':
            return base - q * np.sqrt(u)
        return base - q * u
    best = None
    for branch in ('sqrt_plus', 'sqrt_minus', 'leak'):
        for start in ([2.0, 0.5, 0.1], [3.0, 0.8, 0.3], [1.0, 0.2, 0.5]):
            result = least_squares(lambda p: pred(p, branch) - y, start,
                                   bounds=([0.0, 0.0, 0.0], [20.0, 1.0, 20.0]),
                                   max_nfev=5000)
            err = float(np.mean((pred(result.x, branch) - y) ** 2))
            if best is None or err < best[0]:
                best = (err, result.x.copy(), branch)
    _PARAMS, _BRANCH = best[1], best[2]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row).')
    vmax, rho, q = _PARAMS
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if _BRANCH == 'sqrt_plus':
        return float(base + q * math.sqrt(u))
    if _BRANCH == 'sqrt_minus':
        return float(base - q * math.sqrt(u))
    return float(base - q * u)
