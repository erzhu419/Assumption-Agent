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
    sink = s / (1.0 + s)
    candidates = []
    for branch in ('sqrt', 'linear'):
        term = np.sqrt(np.maximum(u, 0.0)) if branch == 'sqrt' else u
        def residual(p):
            return p[0] * (1.0 - p[1] * sink) * base + p[2] * term - y
        result = least_squares(residual, [5.0, 0.7, 0.5], max_nfev=10000)
        candidates.append((np.dot(result.fun, result.fun), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda c: c[0])

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    term = np.sqrt(max(u, 0.0)) if _branch == 'sqrt' else u
    return float(_params[0] * (1.0 - _params[1] * s / (1.0 + s)) * u / (1.2 + u) + _params[2] * term)
