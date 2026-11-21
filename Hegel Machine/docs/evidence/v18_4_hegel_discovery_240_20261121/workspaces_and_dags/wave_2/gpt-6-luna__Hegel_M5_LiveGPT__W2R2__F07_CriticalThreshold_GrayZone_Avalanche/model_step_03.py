import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(p, r):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(r['u_now'])
    s = float(r['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    lower = np.array([0.01, 0.0, -5.0, 0.0, 2.0])
    upper = np.array([30.0, 0.999, 5.0, 10.0, 3.5])
    best = None
    starts = [
        [6.5, 0.70, 0.2, 2.0, 2.55],
        [5.0, 0.65, 1.0, 3.0, 2.65],
        [8.0, 0.75, -0.5, 1.0, 2.4],
        [4.0, 0.5, 0.0, 4.0, 3.0],
    ]
    for start in starts:
        res = least_squares(residual, np.array(start, dtype=float), bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_PARAMS, row))