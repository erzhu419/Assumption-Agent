import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    flux = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - ys
    starts = ([4.0, 1.0, 0.5, 1.0], [2.0, 0.2, 0.7, 2.0], [8.0, 2.0, 0.3, 0.0])
    results = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 1.0, 20.0]), max_nfev=20000) for x0 in starts]
    _params = min(results, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))