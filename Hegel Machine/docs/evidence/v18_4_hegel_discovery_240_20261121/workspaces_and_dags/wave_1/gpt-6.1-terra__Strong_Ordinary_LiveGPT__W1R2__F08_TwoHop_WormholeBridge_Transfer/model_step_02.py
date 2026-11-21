import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def fit(train_rows):
    global _params, _branch
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = base * s / (1.0 + s)
    candidates = []
    for branch in ('sqrt', 'linear'):
        term = np.sqrt(u) if branch == 'sqrt' else u
        def residual(p):
            return p[0] * base - p[1] * sink + p[2] * term - y
        result = least_squares(residual, [4.0, 2.8, 0.0])
        candidates.append((np.sum(result.fun ** 2), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    term = math.sqrt(u) if _branch == 'sqrt' else u
    return float(_params[0] * base - _params[1] * base * s / (1.0 + s) + _params[2] * term)