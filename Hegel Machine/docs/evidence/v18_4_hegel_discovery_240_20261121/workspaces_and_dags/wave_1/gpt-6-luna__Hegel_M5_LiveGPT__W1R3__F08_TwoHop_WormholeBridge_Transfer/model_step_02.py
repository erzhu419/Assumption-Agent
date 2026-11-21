import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, params):
    vmax, rho, c_loc = params
    u = max(0.0, float(row['u_now']))  # tau_scale is fixed at 1
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.asarray([_predict_with(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([6.5, 0.7, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(row, _params))