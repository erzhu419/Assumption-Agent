import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    def residuals(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residuals, [6.5, 0.5, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_value(row, _params))