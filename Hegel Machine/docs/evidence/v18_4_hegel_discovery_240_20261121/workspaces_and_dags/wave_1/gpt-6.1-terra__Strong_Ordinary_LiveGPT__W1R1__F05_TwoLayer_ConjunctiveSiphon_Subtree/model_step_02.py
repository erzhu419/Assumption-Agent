import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(row, branch):
    u = row['u_now'] / 4.0
    s = row['s_grad']
    base = u / (1.2 + u)
    return np.array([base, -base * s / (1.0 + s), math.sqrt(u) if branch == 'sqrt' else u])

def fit(train_rows):
    global _model, _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for branch in ('sqrt', 'linear'):
        X = np.array([_features(r, branch) for r in train_rows])
        result = least_squares(lambda p: X @ p - y, np.array([4.0, 3.0, 0.0]))
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_features(row, _model) @ _params)
