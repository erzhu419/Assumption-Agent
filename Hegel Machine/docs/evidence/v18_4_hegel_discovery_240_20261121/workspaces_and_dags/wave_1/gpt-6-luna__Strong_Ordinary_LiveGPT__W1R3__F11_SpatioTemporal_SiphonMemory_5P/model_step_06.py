import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    j0, beta, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_with(p, r) for r in rows]) - y
    starts = [[1.0, 0.5, 0.5, 0.0], [2.0, 1.0, 0.7, 0.5], [1.0, 2.0, 0.3, -0.5]]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 100.0, 1.0, 10.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(_params, row))