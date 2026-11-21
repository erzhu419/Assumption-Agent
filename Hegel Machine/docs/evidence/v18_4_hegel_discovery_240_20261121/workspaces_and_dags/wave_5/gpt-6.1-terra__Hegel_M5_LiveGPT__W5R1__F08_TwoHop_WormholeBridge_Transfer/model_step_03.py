import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _model(p, u, g, kind):
    a, b, c, d = np.exp(np.clip(p[:4], -15, 15))
    if kind == 0:
        base = u / (1.0 + b * u) ** d
    else:
        base = np.log1p(b * u) ** d
    return a * base / (1.0 + c * g)

def fit(train_rows):
    global _params, _kind
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    g = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for kind in (0, 1):
        for b0 in (0.2, 1.0, 4.0):
            p0 = np.log([5.0, b0, 0.5, 1.0])
            result = least_squares(lambda p: _model(p, u, g, kind) - y, p0, max_nfev=3000)
            error = np.sum(result.fun ** 2)
            if best is None or error < best[0]:
                best = (error, kind, result.x)
    _, _kind, _params = best

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad']), _kind))
