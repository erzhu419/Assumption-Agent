import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    a, b, c, d, e = p
    return a*u/(b+u) + c*u/(d+u+e*s)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for start in ([3, 1, 4, 1, 1], [1, 0.3, 7, 2, 2], [5, 2, 3, 0.5, 1], [0.5, 1, 8, 1, 1]):
        result = least_squares(lambda p: _model(p, u, s)-y, start, bounds=([0, 1e-6, 0, 1e-6, 0], [100, 100, 100, 100, 100]), max_nfev=5000)
        loss = np.sum(result.fun**2)
        if best is None or loss < best[0]:
            best = (loss, result.x)
    _params = best[1]

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
