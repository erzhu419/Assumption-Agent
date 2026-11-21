import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(u, s, k, mode):
    q = u / (k + u)
    g = s / (1.0 + s)
    if mode == 0:
        return np.array([q, math.sqrt(max(u, 0.0)), -q*g])
    if mode == 1:
        return np.array([q, math.sqrt(max(u, 0.0)), -q*g, -u*g])
    if mode == 2:
        return np.array([q, math.sqrt(max(u, 0.0)), -q*g, -q*g*g])
    return np.array([q, math.sqrt(max(u, 0.0)), -q*g, -u*s])

def _fit_one(rows, mode):
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    n = 3 if mode == 0 else 4
    def residual(p):
        k = math.exp(p[0])
        return np.array([_features(a, b, k, mode).dot(p[1:]) for a, b in zip(u, s)]) - y
    best = None
    for k in (0.03, 0.1, 0.4, 1.2):
        p0 = np.array([math.log(k)] + [4.0, 1.0, 2.0] + ([0.5] if n == 4 else []))
        result = least_squares(residual, p0, bounds=([math.log(0.002)] + [-100.0]*n, [math.log(10.0)] + [100.0]*n), max_nfev=1500)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def fit(train_rows):
    global _model
    rows = list(train_rows)
    best = None
    for mode in range(4):
        errors = []
        for i in range(len(rows)):
            p = _fit_one(rows[:i] + rows[i+1:], mode)
            r = rows[i]
            pred = _features(float(r['u_now']), float(r['s_grad']), math.exp(p[0]), mode).dot(p[1:])
            errors.append((pred - float(r['y']))**2)
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, mode)
    mode = best[1]
    _model = (mode, _fit_one(rows, mode))

def predict(row):
    mode, p = _model
    return float(_features(float(row['u_now']), float(row['s_grad']), math.exp(p[0]), mode).dot(p[1:]))