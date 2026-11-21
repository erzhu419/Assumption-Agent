import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _prediction(row, params):
    gain, log_tau, k_eq = params
    tau = math.exp(float(log_tau))
    contrast = float(row['u_now']) - float(row['u_past'])
    return gain * contrast * math.exp(-float(row['H']) / tau) + k_eq * float(row['u_now'])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must be nonempty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(params):
        return np.asarray([_prediction(r, params) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.0, 0.0]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS.tolist()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_prediction(row, _PARAMS))
