import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(rows):
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    h = np.array([float(r['H']) for r in rows])
    return u, s, h

def _design(u, s, h, p, kind):
    tf, ts = np.exp(p[:2])
    q = s / (1.0 + s)
    base = u / (1.2 + u)
    if kind == 0:
        f = -np.expm1(-h / tf)
        g = -np.expm1(-h / ts)
    elif kind == 1:
        f = -np.expm1(-h / tf)
        g = -np.expm1(-h * np.exp(p[2] * q) / ts)
    else:
        f = -np.expm1(-h * np.exp(p[2] * q) / tf)
        g = -np.expm1(-h / ts)
    return np.column_stack((base, base*q, np.sqrt(np.maximum(u, 0)), u*f, u*g, u*q*g))

def fit(train_rows):
    global _model
    u, s, h = _features(train_rows)
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for kind in range(3):
        def residual(p):
            X = _design(u, s, h, p, kind)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            return X @ coef - y
        for tf in (0.3, 2.0, 8.0):
            for ts in (5.0, 15.0, 40.0):
                p0 = [math.log(tf), math.log(ts)] + ([] if kind == 0 else [0.0])
                lo = [math.log(0.03), math.log(1.0)] + ([] if kind == 0 else [-4.0])
                hi = [math.log(20.0), math.log(100.0)] + ([] if kind == 0 else [4.0])
                result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=1500)
                X = _design(u, s, h, result.x, kind)
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                error = float(np.mean((X @ coef - y)**2))
                score = error + kind * 1e-7
                if best is None or score < best[0]:
                    best = (score, kind, result.x.copy(), coef.copy())
    _model = best

def predict(row):
    _, kind, p, coef = _model
    u, s, h = _features([row])
    return float((_design(u, s, h, p, kind) @ coef)[0])