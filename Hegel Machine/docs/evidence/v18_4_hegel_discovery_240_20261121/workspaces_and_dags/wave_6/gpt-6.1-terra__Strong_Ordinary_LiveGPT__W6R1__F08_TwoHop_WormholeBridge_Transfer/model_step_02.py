import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _features(row, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return np.array([base, -base * s / (1.0 + s), math.sqrt(max(0.0, u)) if branch == 'sqrt' else u])

def fit(train_rows):
    global _params, _branch
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for branch in ('sqrt', 'linear'):
        X = np.array([_features(r, branch) for r in train_rows])
        result = least_squares(lambda p: X @ p - y, np.array([5.0, 3.5, 0.0]))
        error = float(np.sum((X @ result.x - y) ** 2))
        if best is None or error < best[0]:
            best = (error, branch, result.x)
    _, _branch, _params = best

def predict(row):
    return float(_features(row, _branch) @ _params)
