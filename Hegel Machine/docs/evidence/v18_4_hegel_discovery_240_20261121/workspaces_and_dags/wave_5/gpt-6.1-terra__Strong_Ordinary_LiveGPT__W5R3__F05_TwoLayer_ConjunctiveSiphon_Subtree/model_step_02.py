import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(rows):
    u = np.array([r['u_now'] / 4.0 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return u, s

def _evaluate(kind, p, u, s):
    f = u / (1.2 + u)
    if kind == 'sqrt':
        return (p[0] - p[1] * s / (1.0 + s)) * f + p[2] * np.sqrt(u)
    if kind == 'leak':
        return (p[0] - p[1] * s / (1.0 + s)) * f - p[2] * u
    if kind == 'hill':
        return p[0] * u**2 / (p[1] + u**2) / (1.0 + p[2] * s) - p[3] * u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * np.sqrt(u)

def fit(train_rows):
    global _model, _params
    u, s = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = [
        ('sqrt', [5.0, 3.5, 0.5], ([0, 0, -np.inf], [np.inf, np.inf, np.inf])),
        ('leak', [5.0, 3.5, 0.1], ([0, 0, 0], [np.inf, np.inf, np.inf])),
        ('hill', [5.0, 1.0, 0.5, 0.1], ([0, 1e-8, 0, 0], [np.inf, np.inf, np.inf, np.inf])),
        ('barrier', [2.0, 1.0, 0.7, 0.5], ([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]))
    ]
    best = float('inf')
    for kind, start, bounds in candidates:
        result = least_squares(lambda p: _evaluate(kind, p, u, s) - y, start, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = np.mean(result.fun**2)
        if error < best:
            best, _model, _params = error, kind, result.x

def predict(row):
    u, s = _features([row])
    return float(_evaluate(_model, _params, u, s)[0])
