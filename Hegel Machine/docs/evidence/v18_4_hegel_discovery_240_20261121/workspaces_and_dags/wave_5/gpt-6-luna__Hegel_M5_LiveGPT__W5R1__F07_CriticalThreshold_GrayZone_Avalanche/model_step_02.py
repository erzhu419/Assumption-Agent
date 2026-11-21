import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = [
        [6.5, 0.70, 0.1, 1.0, 2.58],
        [6.0, 0.70, 0.5, 1.5, 2.50],
        [8.0, 0.72, -0.2, 0.8, 2.68],
    ]
    best = None
    lo = [1.0, 0.05, -2.0, 0.0, 2.0]
    hi = [15.0, 0.95, 2.0, 10.0, 3.2]
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            x0=x0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = np.asarray(best[1], dtype=float)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))