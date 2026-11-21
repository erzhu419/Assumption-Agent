import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    # Saturating present-supply response plus a gradient-dependent bridge flux.
    sat = 1.0 - math.exp(-max(0.0, u) / max(1e-8, _PARAMS[1]))
    return sat, s

def fit(train_rows):
    global _PARAMS
    def pred(p, r):
        u = max(0.0, float(r['u_now']))
        sat = 1.0 - math.exp(-u / max(1e-8, p[1]))
        # Gradient changes the usable flux; the interaction allows the gradient
        # effect to vary with bridge loading.
        return p[0] + p[2]*sat + p[3]*float(r['s_grad']) + p[4]*sat*float(r['s_grad'])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    best = None
    for scale in (0.5, 1.0, 2.0):
        p0 = np.array([0.5, scale, 2.0, -0.5, 0.0])
        res = least_squares(residual, p0, bounds=([-20, 0.03, -20, -10, -10], [20, 20, 20, 10, 10]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    u = max(0.0, float(row['u_now']))
    sat = 1.0 - math.exp(-u / max(1e-8, _PARAMS[1]))
    return float(_PARAMS[0] + _PARAMS[2]*sat + _PARAMS[3]*float(row['s_grad']) + _PARAMS[4]*sat*float(row['s_grad']))
