import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _basis(row, vmax, rho, c):
    u = max(1e-12, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return np.asarray([_basis(r, vmax, rho, c) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.0]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_basis(row, *_params))