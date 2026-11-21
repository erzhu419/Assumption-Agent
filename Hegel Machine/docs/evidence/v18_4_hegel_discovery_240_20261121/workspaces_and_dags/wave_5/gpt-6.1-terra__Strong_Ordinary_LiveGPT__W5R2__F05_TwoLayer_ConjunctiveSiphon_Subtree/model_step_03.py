import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _features(row, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return np.array([base, -base * s / (1.0 + s), math.sqrt(u) if branch == 'sqrt' else u])

def fit(train_rows):
    global _params, _branch
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for branch in ('sqrt', 'linear'):
        X = np.array([_features(r, branch) for r in train_rows])
        initial = np.linalg.lstsq(X, y, rcond=None)[0]
        result = least_squares(lambda p: X @ p - y, initial)
        loss = float(np.sum(result.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, branch, result.x)
    _, _branch, _params = best

def predict(row):
    return float(_features(row, _branch) @ _params)
