import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(p, row):
    gain, log_tau, k_eq = p
    tau = math.exp(log_tau)
    u_n = float(row['u_now']) / 0.004
    u_p = float(row['u_past']) / 0.004
    return gain * (u_n - u_p) * math.exp(-float(row['H']) / tau) + k_eq * u_n

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.asarray([_predict_with(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, np.array([-1.0, math.log(1.0), 0.3]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(_PARAMS, row))