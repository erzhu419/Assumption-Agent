import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _features(row, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    local = math.sqrt(max(0.0, u)) if branch == 'sqrt' else u
    return np.array([base, -base * s / (1.0 + s), local])

def fit(train_rows):
    global _params, _branch
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    for branch in ('sqrt', 'linear'):
        X = np.array([_features(r, branch) for r in train_rows])
        result = least_squares(lambda p: X @ p - y, np.array([6.0, 4.2, 1.0]))
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row) -> float:
    return float(_features(row, _branch) @ _params)
