import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.005

def _features(row):
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    base_factor = (1.0 - s / (1.0 + s)) * (u / (1.2 + u))
    return base_factor, math.sqrt(max(u, 0.0))

def _value(params, row):
    base_factor, sqrt_u = _features(row)
    vmax, rho, c_loc = params
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (float(row['u_now']) / _TAU / (1.2 + float(row['u_now']) / _TAU)) + c_loc * sqrt_u

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_value(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.15]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))