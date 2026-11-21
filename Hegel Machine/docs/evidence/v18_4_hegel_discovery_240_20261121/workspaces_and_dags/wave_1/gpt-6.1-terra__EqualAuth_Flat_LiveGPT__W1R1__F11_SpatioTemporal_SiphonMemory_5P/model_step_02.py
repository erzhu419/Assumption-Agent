import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(row, kind):
    u = max(float(row['u_now']), 0.0)
    x = float(row['x_now'])
    m = float(row['mem_conv'])
    s = max(float(row['s_grad']), 0.0)
    if kind == 0:
        return np.array([1.0, u, m, s])
    if kind == 1:
        return np.array([1.0, math.sqrt(u), m, s])
    if kind == 2:
        return np.array([1.0, u, m, s / (1.0 + s)])
    return np.array([1.0, math.sqrt(u), m, s / (1.0 + s)])

def _value(row, p, kind):
    h = float(row['H'])
    k = p[4]
    z = k * h
    decay = math.exp(-z)
    integral = h if abs(z) < 1e-8 else -math.expm1(-z) / k
    return float(row['x_now']) * decay + integral * float(np.dot(_features(row, kind), p[:4]))

def _fit_kind(rows, kind):
    def residual(p):
        return np.array([_value(r, p, kind) - float(r['y']) for r in rows])
    best = None
    for k0 in (0.1, 1.0, 3.0):
        result = least_squares(residual, [1.0, 0.5, 0.1, -0.2, k0], bounds=([-20, -20, -20, -20, 0.0], [20, 20, 20, 20, 10.0]), max_nfev=3000)
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    return best[1]

def fit(train_rows):
    global _model
    rows = list(train_rows)
    scores = []
    for kind in range(4):
        errors = []
        for i, row in enumerate(rows):
            p = _fit_kind(rows[:i] + rows[i + 1:], kind)
            errors.append(_value(row, p, kind) - float(row['y']))
        scores.append(float(np.mean(np.square(errors))))
    kind = int(np.argmin(scores))
    _model = (kind, _fit_kind(rows, kind))

def predict(row):
    kind, p = _model
    return float(_value(row, p, kind))
