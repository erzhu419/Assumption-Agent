import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(p, u, s, kind):
    if kind == 0:
        return p[0] + p[1] * u / (np.exp(p[2]) + u) + p[3] / (np.exp(p[4]) + s)
    if kind == 1:
        return p[0] + p[1] * u / (np.exp(p[2]) + u) + p[3] * s / (np.exp(p[4]) + s)
    if kind == 2:
        return p[0] + p[1] * u / (np.exp(p[2]) + u) + p[3] * np.log(np.exp(p[4]) + s)
    return (p[0] + p[1]*u + p[2]*s + p[3]*u*s) / (1.0 + np.exp(p[4])*u + np.exp(p[5])*s)

def _fit(rows, kind):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    n = 6 if kind == 3 else 5
    starts = [np.zeros(n), np.array([0., 4., 0., 1., 0.] + ([0.] if n == 6 else [])), np.array([1., 3., 1., -1., -1.] + ([0.] if n == 6 else []))]
    best = None
    for start in starts:
        result = least_squares(lambda p: _value(p, u, s, kind) - y, start, max_nfev=3000, bounds=(-15, 15))
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    scores = []
    for kind in range(4):
        errors = []
        for i, row in enumerate(rows):
            p = _fit(rows[:i] + rows[i+1:], kind)
            errors.append(float(_value(p, row['u_now'], row['s_grad'], kind) - row['y']))
        scores.append(np.mean(np.square(errors)))
    _model = int(np.argmin(scores))
    _params = _fit(rows, _model)

def predict(row):
    return float(_value(_params, row['u_now'], row['s_grad'], _model))