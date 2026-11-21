import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + branch * max(0.0, u - s)

def fit(train_rows):
    global _params
    result = least_squares(
        lambda p: np.array([_value(r, p) - float(r['y']) for r in train_rows]),
        x0=[8.5, 0.14, 0.0],
        bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]),
    )
    _params = result.x
    return _params

def predict(row):
    return float(_value(row, _params))
