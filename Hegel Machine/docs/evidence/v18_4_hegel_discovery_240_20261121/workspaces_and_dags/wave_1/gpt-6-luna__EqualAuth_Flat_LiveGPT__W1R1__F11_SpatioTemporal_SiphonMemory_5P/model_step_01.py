import numpy as np
from scipy.optimize import least_squares
import math

_KEYS = ('u_past', 'u_now', 'x_now', 'mem_conv', 's_grad', 'H')

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _model(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.2]), bounds=([0.0, 0.0, 0.0, 0.0], [20.0, 20.0, 1.0, 20.0]), max_nfev=10000)
    return result.x

def predict(row):
    return float(_model(_params, row))

_params = np.array([1.0, 0.5, 0.5, 0.2])