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
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    specs = [
        ('sqrt', [4.0, 1.0, 0.0], ([0.0, 0.0, -10.0], [100.0, 3.0, 10.0])),
        ('leak', [4.0, 1.0, 0.1], ([0.0, 0.0, 0.0], [100.0, 3.0, 10.0])),
        ('hill', [4.0, 1.0, 1.0, 0.1], ([0.0, 0.00001, 0.0, 0.0], [100.0, 100.0, 100.0, 10.0])),
        ('barrier', [2.0, 1.0, 1.0, 0.0], ([0.0, 0.0, 0.0, -10.0], [100.0, 30.0, 3.0, 10.0]))
    ]
    for kind, start, bounds in specs:
        def residual(p):
            return np.array([_value(kind, p, r) for r in train_rows]) - y
        result = least_squares(residual, start, bounds=bounds, max_nfev=12000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        candidates.append((float(np.mean(result.fun ** 2)), kind, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_value(_model, _params, row))