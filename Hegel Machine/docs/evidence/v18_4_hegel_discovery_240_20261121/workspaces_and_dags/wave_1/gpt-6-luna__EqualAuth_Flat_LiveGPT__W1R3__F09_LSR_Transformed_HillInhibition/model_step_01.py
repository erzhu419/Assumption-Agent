import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.1
_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 5.0, 0.2, 0.01]), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))