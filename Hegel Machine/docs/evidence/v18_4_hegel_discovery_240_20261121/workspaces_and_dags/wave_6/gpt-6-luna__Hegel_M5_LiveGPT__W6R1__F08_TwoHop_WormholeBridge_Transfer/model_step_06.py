import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * np.sqrt(max(un, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5]), max_nfev=20000)
    _PARAMS = result.x
    return result.x

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))