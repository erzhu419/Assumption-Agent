import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(name, p, row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    if name == 'sqrt':
        return (p[0] + p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * math.sqrt(max(0.0, u))
    if name == 'linear':
        return (p[0] + p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * u
    if name == 'hill':
        return p[0] * u * u / (p[1] + u * u) / (1.0 + p[2] * s) - p[3] * u
    return p[0] * u * math.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    candidates = []
    specs = [
        ('sqrt', [[5.0, -3.5, 0.5]], ([-100, -100, -100], [100, 100, 100])),
        ('linear', [[5.0, -3.5, -0.1]], ([-100, -100, -100], [100, 100, 100])),
        ('hill', [[5.0, 1.0, 0.5, 0.1], [10.0, 4.0, 1.0, 0.5]], ([0, 1e-6, 0, 0], [100, 100, 100, 100])),
        ('barrier', [[2.0, 1.0, 0.7, 1.0], [4.0, 3.0, 0.7, 0.0]], ([0, 0, 0, -100], [100, 100, 1, 100]))
    ]
    for name, starts, bounds in specs:
        for start in starts:
            def residual(p):
                return np.array([_value(name, p, r) for r in rows]) - y
            result = least_squares(residual, start, bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            candidates.append((float(np.dot(result.fun, result.fun)), name, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_value(_model, _params, row))