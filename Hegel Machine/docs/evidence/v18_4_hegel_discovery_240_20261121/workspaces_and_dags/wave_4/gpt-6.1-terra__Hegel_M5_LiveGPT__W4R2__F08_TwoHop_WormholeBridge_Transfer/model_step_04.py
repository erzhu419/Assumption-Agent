import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - row['y'] for row in train_rows])
    _params = least_squares(residual, [5.0, 0.7, 0.5]).x

def _value(row, p):
    vmax, rho, c_loc = p
    u = row['u_now']
    s = row['s_grad']
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)

def predict(row):
    return float(_value(row, _params))