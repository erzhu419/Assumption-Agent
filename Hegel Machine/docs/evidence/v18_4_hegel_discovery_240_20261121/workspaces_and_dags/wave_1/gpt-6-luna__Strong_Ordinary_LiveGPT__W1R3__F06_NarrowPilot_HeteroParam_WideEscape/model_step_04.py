import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(0.0, un))

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(row, _FITTED_PARAMETERS))

_FITTED_PARAMETERS = np.array([5.0, 0.5, 0.5])

def fit(train_rows):
    global _FITTED_PARAMETERS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.5]), max_nfev=20000)
    _FITTED_PARAMETERS = result.x
    return result.x
