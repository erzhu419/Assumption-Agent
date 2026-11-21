import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [7.0, 0.7, 0.5], max_nfev=10000)
    _params = result.x

def predict(row):
    return float(_value(row, _params))
