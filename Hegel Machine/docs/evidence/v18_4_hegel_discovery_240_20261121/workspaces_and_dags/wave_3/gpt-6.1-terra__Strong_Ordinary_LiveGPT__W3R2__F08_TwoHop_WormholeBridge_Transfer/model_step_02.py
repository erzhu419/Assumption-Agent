import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(max(0.0, u)), u

def fit(train_rows):
    global _params
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows])
    # Linear reparameterization of vmax, vmax*rho and the branch coefficient.
    candidates = []
    for branch in (2, 3):
        A = np.column_stack((X[:, 0], -X[:, 0] * X[:, 1], X[:, branch]))
        result = least_squares(lambda p: A @ p - y, np.ones(3))
        candidates.append((np.sum(result.fun ** 2), branch, result.x))
    _, branch, p = min(candidates, key=lambda item: item[0])
    _params = (branch, p)

def predict(row):
    f, g, root, u = _features(row)
    branch, p = _params
    return float(p[0] * f - p[1] * f * g + p[2] * (root if branch == 2 else u))
