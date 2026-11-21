import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    H = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(logp):
        E, eta = np.exp(logp)
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y
    result = least_squares(residual, np.log([0.45, 1.0]), max_nfev=10000)
    _params = tuple(np.exp(result.x))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    E, eta = _params
    u = float(row['u_now'])
    H = float(row['H'])
    return float((u / E) * (1.0 - math.exp(-E * H / eta)))