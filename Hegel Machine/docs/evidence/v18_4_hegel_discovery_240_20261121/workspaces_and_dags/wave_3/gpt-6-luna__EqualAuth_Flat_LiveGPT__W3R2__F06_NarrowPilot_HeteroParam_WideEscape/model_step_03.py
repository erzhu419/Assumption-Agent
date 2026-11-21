import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    a, k, c, d = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = a * u / (k + u) + c * math.sqrt(u)
    return base * math.exp(d * (1.0 - s))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.8, 0.4, 0.5]), bounds=([0.0, 1e-5, 0.0, 0.0], [20.0, 20.0, 20.0, 5.0]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_value(row, _fitted_parameters))