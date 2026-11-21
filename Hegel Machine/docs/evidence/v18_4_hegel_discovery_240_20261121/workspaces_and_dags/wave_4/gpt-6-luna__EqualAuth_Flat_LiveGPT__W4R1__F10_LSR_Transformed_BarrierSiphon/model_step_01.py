import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return u, s

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    inhibition = 1.0 - rho * s / (1.0 + s)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * inhibition + c_sqrt * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5, 0.5]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(row, fit([dict(row, y=0.0)])))