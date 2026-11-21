import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _eval(p, u, s, branch):
    vmax, rho, c = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + c * (np.sqrt(u) if branch == 'sqrt' else u)

def fit(train_rows):
    global _params, _branch
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for branch in ('sqrt', 'linear'):
        result = least_squares(lambda p: _eval(p, u, s, branch) - y,
                               [7.0, 0.7, 0.1],
                               bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
        candidates.append((np.sum(result.fun ** 2), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_eval(_params, float(row['u_now']), float(row['s_grad']), _branch))
