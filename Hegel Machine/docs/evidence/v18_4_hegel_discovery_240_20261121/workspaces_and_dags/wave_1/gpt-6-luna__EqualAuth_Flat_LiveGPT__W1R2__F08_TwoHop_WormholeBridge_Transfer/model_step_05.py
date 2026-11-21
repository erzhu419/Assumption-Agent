import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _value(row, p, branch):
    vmax, rho, c = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt_plus':
        return base + c * math.sqrt(u)
    if branch == 'sqrt_minus':
        return base - c * math.sqrt(u)
    return base - c * u

def fit(train_rows):
    global _MODEL
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    best = None
    for branch in ('sqrt_plus', 'sqrt_minus', 'linear_leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in rows]) - y
        result = least_squares(residual, x0=np.array([4.0, 0.7, 0.2]),
                               bounds=([0.0, 0.0, 0.0], [20.0, 1.0, 10.0]),
                               max_nfev=20000)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, branch, result.x.copy())
    _MODEL = (best[1], best[2])
    return _MODEL

def predict(row) -> float:
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _MODEL[1], _MODEL[0]))