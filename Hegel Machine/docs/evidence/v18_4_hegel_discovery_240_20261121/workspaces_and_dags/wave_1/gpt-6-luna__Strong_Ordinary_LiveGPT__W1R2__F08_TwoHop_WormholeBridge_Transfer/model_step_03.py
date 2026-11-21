import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u = max(u, 0.0)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_branch * np.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 0.7, 0.1]), bounds=([0.0, 0.0, -20.0], [100.0, 2.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))