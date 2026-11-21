import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _features(row, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    extra = math.sqrt(u) if branch == 'sqrt' else u
    return np.array([base, -base * s / (1.0 + s), extra])

def fit(train_rows):
    global _params, _branch
    y = np.array([float(row['y']) for row in train_rows])
    candidates = []
    for branch in ('sqrt', 'linear'):
        X = np.array([_features(row, branch) for row in train_rows])
        result = least_squares(lambda p: X @ p - y, np.array([5.0, 3.5, 0.0]))
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_features(row, _branch) @ _params)
