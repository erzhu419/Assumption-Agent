import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _eval(p, u, s, kind):
    a, k, b, c, q = p
    base = a * u / (k + u) + b * np.sqrt(np.maximum(u, 0.0))
    if kind == 0:
        sink = c * u / (q + u) * s / (1.0 + s)
    elif kind == 1:
        sink = c * u / (q + u) * s
    elif kind == 2:
        sink = c * np.sqrt(np.maximum(u, 0.0)) * s / (q + s)
    else:
        sink = c * u * s / ((q + u) * (1.0 + s))
    return base - sink

def _fit(u, s, y, kind):
    best = None
    for k in (1.0, 5.0, 15.0):
        for q in (1.0, 8.0):
            result = least_squares(lambda p: _eval(p, u, s, kind) - y,
                [4.0, k, 0.5, 3.0, q],
                bounds=([0.0, 0.05, 0.0, 0.0, 0.05], [40.0, 100.0, 10.0, 100.0, 100.0]),
                max_nfev=1500)
            err = np.sum(result.fun ** 2)
            if best is None or err < best[0]:
                best = (err, result.x)
    return best[1]

def fit(train_rows):
    global _params, _kind
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    scores = []
    for kind in (0, 1, 2):
        errors = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            p = _fit(u[mask], s[mask], y[mask], kind)
            errors.append(float(_eval(p, u[i], s[i], kind) - y[i]))
        scores.append(np.mean(np.square(errors)))
    _kind = int(np.argmin(scores))
    _params = _fit(u, s, y, _kind)

def predict(row):
    return float(_eval(_params, float(row['u_now']), float(row['s_grad']), _kind))