import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    lower = np.array([0.01, 0.0, -5.0, 0.0, 2.36])
    upper = np.array([30.0, 0.999, 5.0, 10.0, 4.5])
    starts = [
        [6.5, 0.70, 0.1, 1.0, 2.55],
        [5.0, 0.70, 0.8, 2.0, 2.65],
        [8.0, 0.70, -0.5, 0.5, 2.45],
        [6.0, 0.68, 0.3, 3.0, 2.8],
    ]
    best = None
    for start in starts:
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            np.clip(np.asarray(start, dtype=float), lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=30000, xtol=1e-13, ftol=1e-13, gtol=1e-13
        )
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))