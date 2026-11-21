import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u)

def fit(train_rows):
    global _params
    result = least_squares(lambda p: np.array([_value(r, p) - float(r['y']) for r in train_rows]), [6.0, 0.2, 0.5], bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    _params = result.x

def predict(row):
    return float(_value(row, _params))