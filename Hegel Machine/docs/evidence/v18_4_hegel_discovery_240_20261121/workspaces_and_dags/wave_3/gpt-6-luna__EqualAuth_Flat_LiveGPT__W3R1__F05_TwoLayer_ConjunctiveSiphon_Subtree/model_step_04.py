import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _pred(p, row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    H = float(row['H'])
    # Saturating local input response, attenuated by two conjunctive
    # sink-gradient layers, with a smooth horizon gain.
    a, k1, k2, h = p
    base = a * math.sqrt(u) / (1.0 + k1*s) / (1.0 + k2*s)
    return base * (1.0 + h*H)

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_pred(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([20.0, 0.2, 0.2, 0.0]),
                           bounds=([0.0, 0.0, 0.0, -1.0], [1000.0, 100.0, 100.0, 10.0]),
                           max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('call fit(train_rows) before predict(row)')
    return float(_pred(_params, row))