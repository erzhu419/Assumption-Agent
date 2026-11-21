import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    result = least_squares(
        lambda p: np.array([_value(r, p) - float(r['y']) for r in train_rows]),
        x0=[6.5, 0.5, 0.1],
        bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    _params = result.x

def predict(row):
    return float(_value(row, _params))
