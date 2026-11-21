import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    u = np.array([r['u_now'] / 0.05 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    candidates = []
    for branch in ('sqrt', 'leak'):
        term = np.sqrt(u) if branch == 'sqrt' else u
        def residual(p):
            return p[0] * (1.0 - p[1] * sink) * base + p[2] * term - y
        result = least_squares(residual, [3.0, 0.7, 0.1], max_nfev=5000)
        candidates.append((float(np.dot(result.fun, result.fun)), branch, result.x))
    _, branch, params = min(candidates, key=lambda x: x[0])
    _model = (branch, params)

def predict(row):
    branch, p = _model
    u = row['u_now'] / 0.05
    s = row['s_grad']
    term = math.sqrt(u) if branch == 'sqrt' else u
    return float(p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * term)
