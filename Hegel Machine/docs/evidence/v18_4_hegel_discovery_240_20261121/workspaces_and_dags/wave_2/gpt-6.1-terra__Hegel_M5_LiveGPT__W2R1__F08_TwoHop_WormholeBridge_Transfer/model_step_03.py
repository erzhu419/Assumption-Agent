import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _eval(p, u, s, kind):
    a, b, c, d = np.exp(np.clip(p, -15, 15))
    if kind == 0:
        return a * u / (1.0 + b * u + c * s) + d
    if kind == 1:
        return a * u / (1.0 + b * u) + c / (1.0 + d * s)
    return a * u / (1.0 + b * u) * (1.0 + c / (1.0 + d * s))

def _fit(u, s, y, kind):
    starts = [np.log([3.0, 0.7, 1.0, 0.1]), np.log([5.0, 1.0, 2.0, 1.0]), np.log([2.0, 0.4, 0.3, 2.0])]
    best = None
    for start in starts:
        r = least_squares(lambda p: _eval(p, u, s, kind) - y, start, max_nfev=2000)
        loss = np.sum(r.fun ** 2)
        if best is None or loss < best[0]:
            best = (loss, r.x)
    return best[1]

def fit(train_rows):
    global _params, _kind
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    scores = []
    for kind in range(3):
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