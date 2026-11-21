import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    vmax, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_sqrt * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - ys
    best = None
    for start in ([3.0, 0.5, 0.0], [5.0, 0.8, -0.2], [2.0, 0.2, 0.2]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))