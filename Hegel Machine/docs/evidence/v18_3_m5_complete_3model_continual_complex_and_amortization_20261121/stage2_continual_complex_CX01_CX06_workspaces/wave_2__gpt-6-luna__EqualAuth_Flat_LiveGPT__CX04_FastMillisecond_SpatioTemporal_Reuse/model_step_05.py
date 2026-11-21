import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.005
_P = None

def _predict_with(p, row):
    vmax, rho, c_loc = p
    u_n = row['u_now'] / _TAU
    base = vmax * (1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _P
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.0]), max_nfev=3000)
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_predict_with(_P, row))