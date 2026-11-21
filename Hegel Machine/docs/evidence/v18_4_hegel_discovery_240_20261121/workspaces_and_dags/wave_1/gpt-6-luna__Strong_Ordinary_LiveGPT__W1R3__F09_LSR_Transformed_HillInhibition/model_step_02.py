import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.1
_PARAMS = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n * u_n / (k_half + u_n * u_n)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[3.0, 1.0, 0.2, 0.01], bounds=([0.0, 1e-8, 0.0, 0.0], [100.0, 1e4, 100.0, 100.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))