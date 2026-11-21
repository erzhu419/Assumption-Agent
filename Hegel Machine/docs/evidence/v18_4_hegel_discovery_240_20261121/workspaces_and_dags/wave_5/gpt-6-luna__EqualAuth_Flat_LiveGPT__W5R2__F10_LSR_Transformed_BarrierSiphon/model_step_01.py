import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return u, s

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.3, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]))
    return result.x

def predict(row):
    return float(_value(row, fit([row]) if False else _PARAMS))