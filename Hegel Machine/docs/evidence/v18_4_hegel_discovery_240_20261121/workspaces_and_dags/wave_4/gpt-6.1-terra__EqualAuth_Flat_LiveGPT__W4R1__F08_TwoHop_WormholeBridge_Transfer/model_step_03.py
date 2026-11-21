import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _features(rows):
    return np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows], dtype=float)

def _eval(p, X, kind):
    u, s = X[:, 0], X[:, 1]
    if kind == 0:
        # Saturating response with a gradient-dependent numerator and denominator.
        a, b, c, d, e, f = p
        return (a + b*u + c*s + d*u*s) / (1.0 + e*u + f*s)
    if kind == 1:
        # Two saturating transfer contributions, with gradient modulation.
        a, b, c, d, e, f = p
        return a*u/(1.0+b*u) + c*u/(1.0+d*u+e*s) + f
    # Smooth saturation with a gradient-dependent rate and amplitude.
    a, b, c, d, e, f = p
    return a*(1.0-np.exp(-b*u))/(1.0+c*s) + d*u/(1.0+e*u+f*s)

def _fit_one(X, y, kind):
    if kind == 0:
        lo = [-20, -20, -20, -20, 0, 0]
        hi = [20, 20, 20, 20, 20, 20]
        starts = [[0, 3, 0, 0, 0.5, 0.5], [0, 5, 0, 0, 1, 1]]
    elif kind == 1:
        lo = [-20, 0, -20, 0, 0, -20]
        hi = [20, 20, 20, 20, 20, 20]
        starts = [[2, 0.5, 2, 0.5, 0.5, 0], [4, 1, 2, 1, 1, 0]]
    else:
        lo = [-20, 0, 0, -20, 0, 0]
        hi = [20, 20, 20, 20, 20, 20]
        starts = [[3, 1, 0.5, 2, 1, 0.5], [2, 0.5, 1, 4, 1, 1]]
    best = None
    for start in starts:
        res = least_squares(lambda p: _eval(p, X, kind)-y, start, bounds=(lo, hi), max_nfev=3000)
        loss = np.mean(res.fun**2)
        if best is None or loss < best[0]:
            best = (loss, res.x)
    return best

def fit(train_rows):
    global _params, _kind
    X = _features(train_rows)
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    for kind in range(3):
        loss, p = _fit_one(X, y, kind)
        candidates.append((loss, kind, p))
    _, _kind, _params = min(candidates, key=lambda z: z[0])

def predict(row):
    return float(_eval(_params, _features([row]), _kind)[0])