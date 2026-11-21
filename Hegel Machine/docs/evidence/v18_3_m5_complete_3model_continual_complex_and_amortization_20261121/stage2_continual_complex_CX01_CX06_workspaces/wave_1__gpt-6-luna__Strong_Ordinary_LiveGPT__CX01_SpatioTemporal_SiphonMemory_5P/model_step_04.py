import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _predict_params(row, p, branch):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic tau scale is 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    if branch == 'sqrt':
        return base + c * math.sqrt(max(u_n, 0.0))
    return base - c * u_n

def fit(train_rows):
    global _PARAMS, _BRANCH
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.array([_predict_params(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, x0=np.array([4.0, 0.7, 0.2]),
                               bounds=([0.0, 0.0, -10.0], [30.0, 5.0, 10.0]),
                               max_nfev=20000)
        candidates.append((float(np.mean(residual(result.x)**2)), branch, result.x))
    _, _BRANCH, _PARAMS = min(candidates, key=lambda item: item[0])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _PARAMS, _BRANCH))