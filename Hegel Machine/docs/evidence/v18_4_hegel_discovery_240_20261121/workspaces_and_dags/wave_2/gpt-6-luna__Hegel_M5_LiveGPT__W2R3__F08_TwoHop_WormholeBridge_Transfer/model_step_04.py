import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - ys
    starts = ([3.0, 0.7, 0.1], [5.0, 0.5, 0.0], [2.0, 0.9, 0.5])
    candidates = [least_squares(residual, x0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0])) for x0 in starts]
    best = min(candidates, key=lambda result: float(np.dot(result.fun, result.fun)))
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_params, row))