import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    targets = np.asarray([float(row['y']) for row in train_rows], dtype=float)
    result = least_squares(lambda p: np.asarray([_model(row, p) for row in train_rows]) - targets, x0=[6.5, 0.70, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x

def predict(row):
    return float(_model(row, _params))
