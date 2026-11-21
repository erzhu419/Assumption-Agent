import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    def features(row, branch):
        u = float(row['u_now'])
        s = float(row['s_grad'])
        base = u / (1.2 + u)
        return np.array([base, -base * s / (1.0 + s), math.sqrt(u) if branch == 'sqrt' else u])
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    for branch in ('sqrt', 'linear'):
        X = np.array([features(r, branch) for r in train_rows])
        result = least_squares(lambda p: X @ p - y, np.array([6.0, 4.2, 0.0]))
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, branch, params = min(candidates, key=lambda item: item[0])
    _model = (branch, params)

def predict(row):
    branch, p = _model
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    term = math.sqrt(u) if branch == 'sqrt' else u
    return float(p[0] * base - p[1] * base * s / (1.0 + s) + p[2] * term)
