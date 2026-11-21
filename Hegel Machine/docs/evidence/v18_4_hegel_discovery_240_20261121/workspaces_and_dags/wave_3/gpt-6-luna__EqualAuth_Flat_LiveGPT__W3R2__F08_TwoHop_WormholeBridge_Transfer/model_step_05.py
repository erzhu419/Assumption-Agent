import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p, branch):
    vmax, rho, q = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt':
        return base + q * math.sqrt(u)
    return base - q * u

def fit(train_rows):
    global _P
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.array([_value(r, p, branch) - float(r['y']) for r in train_rows])
        for start in ([3.0, 0.5, 0.5], [5.0, 0.7, 0.1], [2.0, 0.2, 1.0]):
            res = least_squares(residual, np.array(start, dtype=float), bounds=([-100.0, -5.0, -100.0], [100.0, 5.0, 100.0]), max_nfev=20000)
            score = float(np.mean(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, res.x, branch)
    _P = (best[1].copy(), best[2])
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit must be called first')
    return float(_value(row, _P[0], _P[1]))