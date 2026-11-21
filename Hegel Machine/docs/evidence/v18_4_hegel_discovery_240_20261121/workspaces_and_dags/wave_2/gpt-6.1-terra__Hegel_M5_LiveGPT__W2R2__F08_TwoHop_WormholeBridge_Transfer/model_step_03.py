import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(p, u, s, family, mode):
    a, b, c, d = p[:4]
    z = u / (1.0 + c * s) if mode == 1 else u
    if family == 0:
        f = np.log1p(b * z) / b
    elif family == 1:
        f = -np.expm1(-b * z) / b
    else:
        f = np.expm1(d * np.log1p(b * z)) / (b * d)
    if mode == 0:
        f = f / np.power(1.0 + c * s, d if family != 2 else p[4])
    elif mode == 2:
        f = f - d * s if family != 2 else f - p[4] * s
    return a * f

def _fit_one(u, s, y, family, mode):
    n = 5 if family == 2 and mode != 1 else 4
    lower = [0.001, 0.0001, 0.0, 0.02] + ([0.0] if n == 5 else [])
    upper = [50.0, 100.0, 20.0, 3.0] + ([3.0] if n == 5 else [])
    best = None
    for b0, c0 in [(0.3, 0.2), (1.0, 1.0), (3.0, 0.5)]:
        p0 = [3.0, b0, c0, 0.5] + ([0.5] if n == 5 else [])
        r = least_squares(lambda p: _value(p, u, s, family, mode) - y,
                          p0, bounds=(lower, upper), max_nfev=1500)
        err = np.sum(r.fun ** 2)
        if best is None or err < best[0]:
            best = (err, r.x)
    return best[1]

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = [(f, m) for f in range(3) for m in range(3)]
    best = None
    for family, mode in candidates:
        errors = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            p = _fit_one(u[mask], s[mask], y[mask], family, mode)
            errors.append(float(_value(p, u[i], s[i], family, mode) - y[i]))
        score = np.mean(np.square(errors))
        if best is None or score < best[0]:
            best = (score, family, mode)
    _model = (best[1], best[2])
    _params = _fit_one(u, s, y, *_model)

def predict(row):
    return float(_value(_params, float(row['u_now']), float(row['s_grad']), *_model))