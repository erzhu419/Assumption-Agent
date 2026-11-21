import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, rho, c_loc = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_predict_with(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))