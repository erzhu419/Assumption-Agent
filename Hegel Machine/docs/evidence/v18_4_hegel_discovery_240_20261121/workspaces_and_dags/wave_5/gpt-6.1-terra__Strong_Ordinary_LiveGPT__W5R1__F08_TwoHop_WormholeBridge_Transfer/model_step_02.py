import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = base * s / (1.0 + s)
    candidates = []
    for branch in ('sqrt', 'linear'):
        term = np.sqrt(np.maximum(u, 0.0)) if branch == 'sqrt' else u
        X = np.column_stack((base, -sink, term))
        result = least_squares(lambda p: X @ p - y, np.array([8.0, 5.6, 0.0]))
        candidates.append((np.sum(result.fun ** 2), branch, result.x))
    _, branch, params = min(candidates, key=lambda item: item[0])
    _model = (branch, params)

def predict(row):
    branch, p = _model
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    term = math.sqrt(max(u, 0.0)) if branch == 'sqrt' else u
    return float(p[0] * base - p[1] * base * s / (1.0 + s) + p[2] * term)
