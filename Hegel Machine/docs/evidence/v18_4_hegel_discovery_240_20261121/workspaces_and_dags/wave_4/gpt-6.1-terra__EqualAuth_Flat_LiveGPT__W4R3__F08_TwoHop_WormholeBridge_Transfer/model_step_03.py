import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return p[0] + p[1] * np.log1p(np.exp(p[2]) * u) - p[3] * np.log1p(np.exp(p[4]) * s)

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for start in ([0, 2, 0, 2, 0], [1, 1, 1, 1, 1], [0, 3, -1, 3, -1]):
        result = least_squares(lambda p: _evaluate(p, train_rows) - y, start, bounds=([-100, -100, -8, -100, -8], [100, 100, 8, 100, 8]), max_nfev=10000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, [row])[0])