import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    local = c_loc * math.sqrt(max(u, 0.0))
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + local - avalanche

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.5, 0.3, 1.0, 2.6], [5.0, 0.2, 0.5, 1.5, 2.5], [3.0, 0.8, 0.2, 0.8, 2.7]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, 0.0, 0.0, 1.0], [20.0, 1.0, 10.0, 20.0, 4.0]), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))