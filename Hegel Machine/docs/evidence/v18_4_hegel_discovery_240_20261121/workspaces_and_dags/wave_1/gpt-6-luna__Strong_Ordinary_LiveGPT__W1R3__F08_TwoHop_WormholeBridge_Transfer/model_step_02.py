import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _base(row, vmax, rho):
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))

def fit(train_rows):
    global _PARAMS, _BRANCH
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    candidates = [
        ('sqrt_plus', lambda r, p: _base(r, p[0], p[1]) + p[2] * math.sqrt(max(0.0, float(r['u_now'])))),
        ('sqrt_minus', lambda r, p: _base(r, p[0], p[1]) - p[2] * math.sqrt(max(0.0, float(r['u_now'])))),
        ('linear_leak', lambda r, p: _base(r, p[0], p[1]) - p[2] * max(0.0, float(r['u_now']))),
    ]
    best = None
    for name, fn in candidates:
        def residual(p):
            return np.array([fn(r, p) for r in rows]) - y
        for start in ([5.0, 0.5, 0.5], [8.0, 0.7, 1.0], [4.0, 0.2, 0.1]):
            result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 100.0]), max_nfev=5000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, name, result.x.copy())
    _, _BRANCH, _PARAMS = best
    return {'branch': _BRANCH, 'params': _PARAMS.tolist()}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, rho, c = _PARAMS
    u = max(0.0, float(row['u_now']))
    base = _base(row, vmax, rho)
    if _BRANCH == 'sqrt_plus':
        return float(base + c * math.sqrt(u))
    if _BRANCH == 'sqrt_minus':
        return float(base - c * math.sqrt(u))
    return float(base - c * u)
