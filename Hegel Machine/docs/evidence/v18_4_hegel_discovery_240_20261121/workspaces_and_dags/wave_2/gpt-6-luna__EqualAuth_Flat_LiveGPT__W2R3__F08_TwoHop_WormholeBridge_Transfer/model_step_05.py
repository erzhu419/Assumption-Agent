import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None
_PARAMS = None

def _predict_kind(row, kind, p, branch_sign=1.0):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = p[0] * (1.0 - p[1] * s / (1.0 + s)) * (u / (1.2 + u))
    if kind == 'sqrt':
        return base + branch_sign * p[2] * math.sqrt(u)
    if kind == 'leak':
        return base - p[2] * u
    # Barrier-modulated siphon with open positive square-root branch.
    return p[0] * u * math.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * math.sqrt(u)

def fit(train_rows):
    global _MODEL, _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    best = None
    candidates = [('sqrt', 3, 1.0), ('sqrt', 3, -1.0), ('leak', 3, 1.0), ('barrier', 4, 1.0)]
    for kind, n, sign in candidates:
        def residual(p):
            return np.asarray([_predict_kind(r, kind, p, sign) for r in train_rows]) - ys
        starts = ([3.0, 0.5, 0.2] if n == 3 else [3.0, 0.5, 0.5, 0.1])
        lo = ([0.0, 0.0, 0.0] if n == 3 else [0.0, 0.0, 0.0, 0.0])
        hi = ([20.0, 2.0, 10.0] if n == 3 else [30.0, 20.0, 2.0, 10.0])
        result = least_squares(residual, starts, bounds=(lo, hi), max_nfev=20000)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, kind, sign, result.x.copy())
    _, _MODEL, sign, _PARAMS = best
    if _MODEL == 'sqrt':
        _PARAMS = np.asarray([*_PARAMS, sign])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    if _MODEL == 'sqrt':
        return _predict_kind(row, 'sqrt', _PARAMS[:3], float(_PARAMS[3]))
    return _predict_kind(row, _MODEL, _PARAMS)
