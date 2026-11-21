import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _value(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    local = math.sqrt(u) if branch == 'sqrt' else u
    return (p[0] - p[1] * s / (1.0 + s)) * base + p[2] * local

def fit(train_rows):
    global _params, _branch
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for branch in ('sqrt', 'linear'):
        def residual(p):
            return np.array([_value(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, [5.0, 3.5, 0.0])
        candidates.append((np.dot(result.fun, result.fun), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda c: c[0])

def predict(row):
    return float(_value(row, _params, _branch))
