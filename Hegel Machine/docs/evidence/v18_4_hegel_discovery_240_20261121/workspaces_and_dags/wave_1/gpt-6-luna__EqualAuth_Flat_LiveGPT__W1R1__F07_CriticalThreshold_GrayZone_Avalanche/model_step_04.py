import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    u_n = u  # supplied tau_scale is 1.0
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    starts = [[3.0, 1.0, 0.7, 0.0], [5.0, 2.0, 0.5, -0.5], [1.0, 0.2, 0.9, 0.5]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, -5.0, -10.0], [100.0, 30.0, 10.0, 10.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))