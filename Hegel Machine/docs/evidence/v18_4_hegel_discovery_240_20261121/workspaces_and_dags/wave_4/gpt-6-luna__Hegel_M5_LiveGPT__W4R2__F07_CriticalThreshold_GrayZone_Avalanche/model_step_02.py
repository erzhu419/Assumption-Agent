import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(u, 0.0))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    starts = [
        [6.5, 0.70, 0.1, 1.0, 2.6],
        [5.0, 0.68, 0.5, 2.0, 2.5],
        [8.0, 0.74, -0.3, 0.5, 2.7],
    ]
    best = None
    lo = [1.0, 0.05, -2.0, 0.0, 2.36]
    hi = [15.0, 0.95, 2.0, 10.0, 3.2]
    for x0 in starts:
        res = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - ys,
            x0=np.asarray(x0, dtype=float), bounds=(lo, hi), max_nfev=20000
        )
        loss = float(np.dot(res.fun, res.fun))
        if best is None or loss < best[0]:
            best = (loss, res.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))