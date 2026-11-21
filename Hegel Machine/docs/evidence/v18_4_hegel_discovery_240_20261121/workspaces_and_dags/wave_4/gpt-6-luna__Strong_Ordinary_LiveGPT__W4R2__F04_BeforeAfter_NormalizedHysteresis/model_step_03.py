import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.004
_PARAMS = None

def _predict_params(p, row):
    gain, log_tau, k_eq = p
    tau = math.exp(float(log_tau))
    un = float(row['u_now']) / _TAU_SCALE
    up = float(row['u_past']) / _TAU_SCALE
    H = float(row['H'])
    return gain * (un - up) * math.exp(-H / tau) + k_eq * un

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(p, r) for r in rows]) - y
    best = None
    for tau0 in (0.004, 0.05, 0.5, 5.0):
        p0 = np.array([-1.0, math.log(tau0), 0.2], dtype=float)
        result = least_squares(residual, p0, bounds=([-100.0, math.log(1e-5), -100.0], [100.0, math.log(1e3), 100.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x.copy()
    return {'gain': float(_PARAMS[0]), 'tau_relax': float(math.exp(_PARAMS[1])), 'k_eq': float(_PARAMS[2])}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(_PARAMS, row))