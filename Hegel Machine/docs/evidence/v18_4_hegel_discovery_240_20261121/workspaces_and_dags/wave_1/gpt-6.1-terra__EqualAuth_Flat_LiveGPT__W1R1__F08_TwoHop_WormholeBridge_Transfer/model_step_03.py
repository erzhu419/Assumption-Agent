import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(rows):
    return np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows], dtype=float)

def _evaluate(p, X, family):
    u = np.maximum(X[:, 0], 0.0)
    s = np.maximum(X[:, 1], 0.0)
    b, d = np.exp(np.clip(p[3:5], -8.0, 8.0))
    if family[0] == 'log':
        f = np.log1p(b * u)
    else:
        f = u / (b + u)
    if family[1] == 'log':
        g = np.log1p(d * s) - np.log1p(d)
    elif family[1] == 'rational':
        g = (s - 1.0) / (1.0 + d * s)
    else:
        g = s - 1.0
    return p[0] + p[1] * f + p[2] * g

def _fit_one(X, y, family):
    best = None
    for b in (0.3, 1.0, 3.0):
        p0 = np.array([0.0, 4.0, -1.0, math.log(b), 0.0])
        result = least_squares(lambda p: _evaluate(p, X, family) - y, p0,
                               bounds=([-30, -100, -100, -8, -8],
                                       [30, 100, 100, 8, 8]), max_nfev=2000)
        if best is None or np.sum(result.fun ** 2) < best[0]:
            best = (np.sum(result.fun ** 2), result.x)
    return best[1]

def fit(train_rows):
    global _model, _params
    X = _features(train_rows)
    y = np.array([float(r['y']) for r in train_rows])
    families = [(a, b) for a in ('log', 'saturating')
                for b in ('log', 'rational', 'linear')]
    best = None
    for family in families:
        errors = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            p = _fit_one(X[mask], y[mask], family)
            errors.append(float(_evaluate(p, X[i:i+1], family)[0] - y[i]))
        score = float(np.mean(np.square(errors)))
        if best is None or score < best[0]:
            best = (score, family)
    _model = best[1]
    _params = _fit_one(X, y, _model)

def predict(row):
    return float(_evaluate(_params, _features([row]), _model)[0])