import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def fit(train_rows):
    global _params, _branch
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    candidates = []
    for branch in ('sqrt', 'linear'):
        term = np.sqrt(u) if branch == 'sqrt' else u
        def residual(p):
            return p[0] * (1.0 - p[1] * sink) * base + p[2] * term - y
        bounds = ([0.0, 0.69, -np.inf], [np.inf, 0.72, np.inf])
        if branch == 'linear':
            bounds = ([0.0, 0.69, -np.inf], [np.inf, 0.72, 0.0])
        result = least_squares(residual, [4.0, 0.705, -0.1 if branch == 'linear' else 0.1], bounds=bounds)
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    term = math.sqrt(u) if _branch == 'sqrt' else u
    return float(_params[0] * (1.0 - _params[1] * s / (1.0 + s)) * u / (1.2 + u) + _params[2] * term)
