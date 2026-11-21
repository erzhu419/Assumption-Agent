import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _values(rows):
    return np.array([r['u_now'] for r in rows], dtype=float), np.array([r['s_grad'] for r in rows], dtype=float)

def _eval(kind, p, u, s):
    if kind == 0:
        a, b, c, d, e = p
        return a*u/(b+u) + c*u/(d+u+e*s)
    if kind == 1:
        a, b, c, d, e = p
        return a*u/(b+u) + c*u/((d+u)*(1+e*s))
    if kind == 2:
        a, b, c, d, e = p
        return a*u/((b+u)*(1+c*s)) + d*u/(e+u)
    a, b, c, d, e, f = p
    return a*u/(b+u+c*s) + d*u/(e+u+f*s)

def fit(train_rows):
    global _model, _params
    u, s = _values(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for kind in range(4):
        n = 6 if kind == 3 else 5
        for scale in (0.3, 1.0, 3.0):
            p0 = np.full(n, scale)
            result = least_squares(lambda p: _eval(kind, p, u, s)-y, p0, bounds=(1e-8, 100.0), max_nfev=5000)
            error = np.mean(result.fun**2)
            if best is None or error < best[0]:
                best = (error, kind, result.x)
    _, _model, _params = best

def predict(row):
    return float(_eval(_model, _params, float(row['u_now']), float(row['s_grad'])))