import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _value(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    local = math.sqrt(max(0.0, u)) if branch == 'sqrt' else u
    return p[0] * base + p[1] * base * s / (1.0 + s) + p[2] * local

def fit(train_rows):
    global _params, _branch
    best = None
    for branch in ('sqrt', 'linear'):
        result = least_squares(lambda p: np.array([_value(r, p, branch) - float(r['y']) for r in train_rows]), [5.0, -3.5, 0.1])
        error = float(np.dot(result.fun, result.fun))
        if best is None or error < best[0]:
            best = (error, result.x, branch)
    _, _params, _branch = best

def predict(row):
    return float(_value(row, _params, _branch))