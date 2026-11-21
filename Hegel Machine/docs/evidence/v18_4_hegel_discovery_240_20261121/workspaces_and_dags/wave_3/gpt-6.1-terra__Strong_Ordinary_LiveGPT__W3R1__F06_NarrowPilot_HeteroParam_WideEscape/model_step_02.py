import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(kind, p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    if kind == 'sqrt':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * math.sqrt(max(0.0, u))
    if kind == 'leak':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) - p[2] * u
    if kind == 'hill':
        return p[0] * u * u / (p[1] + u * u) / (1.0 + p[2] * s) - p[3] * u
    return p[0] * u * math.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    candidates = [
        ('sqrt', [5.0, 0.7, 0.5], ([0.0, 0.0, -20.0], [100.0, 2.0, 20.0])),
        ('leak', [5.0, 0.7, 0.1], ([0.0, 0.0, -20.0], [100.0, 2.0, 20.0])),
        ('hill', [5.0, 1.0, 0.5, 0.1], ([0.0, 0.00001, 0.0, -20.0], [100.0, 100.0, 20.0, 20.0])),
        ('barrier', [2.0, 1.0, 0.7, 1.0], ([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 2.0, 20.0]))
    ]
    best = float('inf')
    for kind, start, bounds in candidates:
        def residual(p):
            return np.array([_value(kind, p, r) - float(r['y']) for r in rows])
        result = least_squares(residual, start, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        score = float(np.dot(result.fun, result.fun))
        if score < best:
            best, _model, _params = score, kind, result.x

def predict(row):
    return float(_value(_model, _params, row))