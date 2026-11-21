import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _eval(kind, p, u, s):
    if kind == 'sqrt':
        return p[0] * (1 - p[1] * s / (1 + s)) * u / (1.2 + u) + p[2] * np.sqrt(u)
    if kind == 'linear':
        return p[0] * (1 - p[1] * s / (1 + s)) * u / (1.2 + u) - p[2] * u
    if kind == 'hill':
        return p[0] * u**2 / (p[1] + u**2) / (1 + p[2] * s) - p[3] * u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1 - p[2] * s / (1 + s)) + p[3] * np.sqrt(u)

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = [
        ('sqrt', [7, 0.7, 0.2], ([0, 0, -np.inf], [np.inf, 2, np.inf])),
        ('linear', [7, 0.7, 0.1], ([0, 0, -np.inf], [np.inf, 2, np.inf])),
        ('hill', [6, 1, 1, 0.1], ([0, 1e-8, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])),
        ('barrier', [2, 1, 0.7, 0.1], ([0, 0, 0, -np.inf], [np.inf, np.inf, 2, np.inf]))
    ]
    best = float('inf')
    for kind, initial, bounds in candidates:
        result = least_squares(lambda p: _eval(kind, p, u, s) - y, initial, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = np.sum(result.fun**2)
        if error < best:
            best, _model, _params = error, kind, result.x

def predict(row):
    return float(_eval(_model, _params, float(row['u_now']), float(row['s_grad'])))