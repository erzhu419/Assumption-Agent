import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic tau_scale is 1.0
    source = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return source - leak * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 0.7, 0.1]), bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))