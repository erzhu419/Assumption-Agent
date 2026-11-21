import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [6.5, 0.5, 0.2], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x

def predict(row):
    return float(_model(row, _params))
