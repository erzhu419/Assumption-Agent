import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, rho, c_loc = p
    un = float(row['u_now']) / 4.0
    sg = float(row['s_grad'])
    return vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.2]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))