import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def fit(train_rows):
    global _params, _branch
    u = np.array([r['u_now'] / 4.0 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    candidates = []
    for branch, term in [('sqrt', np.sqrt(u)), ('linear', u)]:
        X = np.column_stack((base, -base * sink, term))
        result = least_squares(lambda p: X @ p - y, np.array([8.0, 5.6, 0.5]))
        candidates.append((np.linalg.norm(result.fun), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    u = row['u_now'] / 4.0
    s = row['s_grad']
    base = u / (1.2 + u)
    term = math.sqrt(u) if _branch == 'sqrt' else u
    return float(_params[0] * base - _params[1] * base * s / (1.0 + s) + _params[2] * term)
