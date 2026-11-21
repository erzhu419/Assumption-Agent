import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(rows, mode, p):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    q = s / (1.0 + s)
    base = u / (1.2 + u)
    t1, t2 = np.exp(p[:2])
    f = -np.expm1(-h / t1)
    g = -np.expm1(-h / t2)
    if mode == 0:
        return np.column_stack((base, base*q, np.sqrt(u), u*f, u*g))
    if mode == 1:
        return np.column_stack((base, base*q, np.sqrt(u), base*f, base*g))
    if mode == 2:
        return np.column_stack((base, base*q, u*f, u*g, u*q*g))
    if mode == 3:
        return np.column_stack((base, base*q, np.sqrt(u), u*f, u*q*f, u*g))
    return np.column_stack((base, base*q, np.sqrt(u), u*f, u*g, u*q*g))

def fit(train_rows):
    global _model
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    candidates = []
    for mode in range(5):
        def residual(p):
            X = _features(rows, mode, p)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            return X @ coef - y
        best = None
        for t1, t2 in [(0.3, 8), (1, 15), (3, 20), (0.1, 3), (5, 30)]:
            result = least_squares(residual, np.log([t1, t2]), bounds=(np.log([0.02, 0.1]), np.log([100.0, 200.0])), max_nfev=1500)
            err = np.linalg.norm(result.fun)
            if best is None or err < best[0]:
                best = (err, result.x)
        X = _features(rows, mode, best[1])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        candidates.append((best[0], mode, best[1], coef))
    _, mode, p, coef = min(candidates, key=lambda item: item[0])
    _model = (mode, p, coef)

def predict(row):
    mode, p, coef = _model
    return float(_features([row], mode, p)[0] @ coef)
