import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _evaluate(p, u, s, kind):
    a, b, c, d, e = p
    b, c = np.exp(b), np.exp(c)
    if kind == 0:
        return a * np.log1p(b * u) / np.power(1.0 + c * s, d) + e
    return a * np.log1p(b * u / np.power(1.0 + c * s, d)) + e

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for kind in (0, 1):
        for start in ([2.0, 0.0, -1.0, 0.5, 0.0], [1.5, 1.0, 0.0, 0.5, 0.0]):
            result = least_squares(lambda p: _evaluate(p, u, s, kind) - y, start, bounds=([-20, -8, -8, -4, -20], [20, 8, 8, 4, 20]), max_nfev=5000)
            err = np.sum(result.fun ** 2)
            if best is None or err < best[0]:
                best = (err, kind, result.x)
    _, _model, _params = best

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['s_grad']), _model))