import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, km, c = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [6.0, 0.8, 1.2, 0.3], bounds=([0.0, 0.0, 1e-6, -10.0], [100.0, 2.0, 100.0, 10.0]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    return float(_value(row, _params))