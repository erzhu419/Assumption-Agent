import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    starts = [
        [5.0, 0.70, 0.0, 2.0, 2.6],
        [4.0, 0.50, 0.2, 1.0, 2.5],
        [8.0, 0.80, -0.5, 3.0, 2.7],
    ]
    lo = [1e-6, 0.0, -10.0, 0.0, 2.0]
    hi = [100.0, 1.5, 10.0, 20.0, 3.2]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_predict_params(r, p) for r in train_rows]) - y,
                            x0=np.asarray(x0, dtype=float), bounds=(lo, hi), max_nfev=20000)
        err = float(np.dot(res.fun, res.fun))
        if best is None or err < best[0]:
            best = (err, res.x)
    _PARAMS = best[1]
    return _PARAMS.tolist()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))