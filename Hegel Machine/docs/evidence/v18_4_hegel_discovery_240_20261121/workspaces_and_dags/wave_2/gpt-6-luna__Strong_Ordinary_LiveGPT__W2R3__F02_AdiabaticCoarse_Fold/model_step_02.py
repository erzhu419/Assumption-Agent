import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    k_ss, a_eff = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return k_ss * u + (x - k_ss * u) * math.exp(-a_eff * H)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        k_ss, a_eff = p
        return np.asarray([
            k_ss * float(r['u_now']) +
            (float(r['x_now']) - k_ss * float(r['u_now'])) *
            math.exp(-a_eff * float(r['H'])) - float(r['y'])
            for r in train_rows
        ])
    result = least_squares(residual, x0=np.array([2.2, 0.02]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params