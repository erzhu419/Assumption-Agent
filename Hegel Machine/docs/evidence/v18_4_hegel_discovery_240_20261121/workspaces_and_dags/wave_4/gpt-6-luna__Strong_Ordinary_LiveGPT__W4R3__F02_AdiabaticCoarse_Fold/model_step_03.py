import numpy as np
from scipy.optimize import least_squares

_params = np.array([0.0, 2.0])

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return p[0] + p[1] * u - y
    result = least_squares(residual, np.array([0.0, 2.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_params[0] + _params[1] * float(row['u_now']))