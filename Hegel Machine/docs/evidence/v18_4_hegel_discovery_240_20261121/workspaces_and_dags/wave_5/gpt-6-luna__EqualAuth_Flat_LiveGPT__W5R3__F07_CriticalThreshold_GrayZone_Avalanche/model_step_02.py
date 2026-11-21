import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    u, s = _features(row)
    vmax, rho, c_loc, k_aval, s_crit = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    starts = [
        [5.0, 0.5, 0.5, 1.0, 2.6],
        [4.0, 0.2, 0.8, 1.5, 2.5],
        [8.0, 0.8, 0.2, 0.8, 2.7],
    ]
    best = None
    for x0 in starts:
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in rows]) - y,
            x0=np.asarray(x0, dtype=float),
            bounds=([0.0, 0.0, 0.0, 0.0, 2.0], [30.0, 1.0, 10.0, 10.0, 3.2]),
            max_nfev=20000,
        )
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))