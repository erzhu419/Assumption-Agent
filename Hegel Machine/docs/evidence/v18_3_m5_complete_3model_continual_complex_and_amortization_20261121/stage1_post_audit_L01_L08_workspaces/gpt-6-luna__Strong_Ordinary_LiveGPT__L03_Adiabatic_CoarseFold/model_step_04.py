import math
import numpy as np
from scipy.optimize import least_squares

_FIT = None

def _value(row, p):
    vmax, rho, c_branch = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_branch * math.sqrt(u)

def fit(train_rows):
    global _FIT
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    # Compare the documented positive/negative square-root branches and linear-leak alternative.
    for kind in ('sqrt', 'leak'):
        def pred(r, p):
            vmax, rho, c = p
            u = max(float(r['u_now']), 0.0)
            s = float(r['s_grad'])
            base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            return base + (c * math.sqrt(u) if kind == 'sqrt' else -c * u)
        def fun(p):
            return np.asarray([pred(r, p) for r in train_rows]) - y
        for sign in (-1.0, 1.0):
            p0 = [8.0, 0.7, sign * 1.0]
            lo = [0.0, 0.0, -20.0] if kind == 'sqrt' else [0.0, 0.0, 0.0]
            hi = [100.0, 1.0, 20.0]
            if kind == 'leak':
                p0[2] = 0.5
            res = least_squares(fun, p0, bounds=(lo, hi), max_nfev=20000)
            score = float(np.mean(fun(res.x) ** 2))
            if best is None or score < best[0]:
                best = (score, kind, res.x)
    _FIT = best[1:]
    return _FIT

def predict(row):
    if _FIT is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    kind, p = _FIT
    vmax, rho, c = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return float(base + (c * math.sqrt(u) if kind == 'sqrt' else -c * u))