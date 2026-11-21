import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return u, s

def _eval(kind, p, u, s):
    if kind == 'sqrt':
        return (p[0] - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * np.sqrt(np.maximum(u, 0.0))
    if kind == 'leak':
        return (p[0] - p[1] * s / (1.0 + s)) * u / (1.2 + u) - p[2] * u
    if kind == 'hill':
        return p[0] * u**2 / (p[1] + u**2) / (1.0 + p[2] * s) - p[3] * u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * np.sqrt(np.maximum(u, 0.0))

def fit(train_rows):
    global _model, _params
    u, s = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for kind in ('sqrt', 'leak'):
        res = least_squares(lambda p: _eval(kind, p, u, s) - y, [5.0, 3.5, 0.5])
        candidates.append((np.sum(res.fun**2), kind, res.x))
    for kind, starts, bounds in (
        ('hill', [[5, 1, 0.5, 0.1], [10, 3, 1, 0.5]], ([0, 1e-8, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])),
        ('barrier', [[2, 1, 0.7, 1], [5, 3, 0.7, 0.5]], ([0, 0, 0, -np.inf], [np.inf, np.inf, 2, np.inf]))):
        for start in starts:
            res = least_squares(lambda p: _eval(kind, p, u, s) - y, start, bounds=bounds, max_nfev=5000)
            candidates.append((np.sum(res.fun**2), kind, res.x))
    _, _model, _params = min(candidates, key=lambda x: x[0])

def predict(row):
    u, s = _features([row])
    return float(_eval(_model, _params, u, s)[0])