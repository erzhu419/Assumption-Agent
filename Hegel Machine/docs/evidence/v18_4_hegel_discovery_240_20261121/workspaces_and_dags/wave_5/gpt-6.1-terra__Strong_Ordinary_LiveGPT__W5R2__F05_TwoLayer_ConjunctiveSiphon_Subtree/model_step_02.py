import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _value(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u)
    return base + p[2] * (math.sqrt(u) if branch == 'sqrt' else u)

def fit(train_rows):
    global _params, _branch
    y = np.array [float(r['y']) for r in train_rows]
    best = None
    for branch in ('sqrt', 'linear'):
        def residual(p):
            return np.array([_value(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, [6.0, 0.7, 0.5], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
        error = float(np.dot(result.fun, result.fun))
        if best is None or error < best[0]:
            best = (error, result.x, branch)
    _, _params, _branch = best

def predict(row):
    return float(_value(row, _params, _branch))