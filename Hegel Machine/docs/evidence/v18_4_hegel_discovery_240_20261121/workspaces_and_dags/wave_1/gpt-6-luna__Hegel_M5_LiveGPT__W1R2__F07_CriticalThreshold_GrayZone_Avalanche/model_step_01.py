import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))  # tau_scale = 1.0
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    p0 = np.array([6.5, 0.70, 0.1])
    lower = np.array([1.0, 0.05, -2.0])
    upper = np.array([15.0, 0.95, 2.0])
    result = least_squares(lambda p: np.asarray([_predict_with(r, p) for r in train_rows]) - y,
                           p0, bounds=(lower, upper))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(row, _params))