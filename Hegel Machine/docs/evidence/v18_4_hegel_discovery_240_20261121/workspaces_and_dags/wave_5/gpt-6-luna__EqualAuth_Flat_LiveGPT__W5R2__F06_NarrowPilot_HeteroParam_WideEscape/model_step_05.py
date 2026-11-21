import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, p):
    vmax, rho, c_loc = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_response(r, p) for r in train_rows]) - y
    best = None
    for start in ([5.0, 0.5, 0.5], [3.0, 0.2, 1.0], [8.0, 0.8, 0.0]):
        result = least_squares(residual, np.asarray(start), bounds=([0.0, -5.0, -10.0], [100.0, 5.0, 10.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _params))