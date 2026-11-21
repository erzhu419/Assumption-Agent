import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    u = np.asarray([float(r['u_now']) for r in rows], dtype=float)
    s = np.asarray([float(r['s_grad']) for r in rows], dtype=float)

    def predict_params(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                + c_loc * np.sqrt(np.maximum(u, 0.0))
                - k_aval * np.maximum(0.0, s - s_crit) * u)

    def residual(p):
        return predict_params(p) - y

    lower = np.array([0.01, 0.0, -5.0, 0.0, 2.50])
    upper = np.array([30.0, 1.0, 5.0, 5.0, 2.68])
    starts = [
        [6.5, 0.70, 0.2, 0.8, 2.59],
        [5.0, 0.65, 0.8, 1.2, 2.55],
        [8.0, 0.75, -0.2, 0.5, 2.65],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, np.asarray(start, dtype=float),
                               bounds=(lower, upper), max_nfev=20000,
                               xtol=1e-13, ftol=1e-13, gtol=1e-13)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1].copy()
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, rho, c_loc, k_aval, s_crit = _PARAMS
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
                 + c_loc * math.sqrt(max(u, 0.0))
                 - k_aval * max(0.0, s - s_crit) * u)