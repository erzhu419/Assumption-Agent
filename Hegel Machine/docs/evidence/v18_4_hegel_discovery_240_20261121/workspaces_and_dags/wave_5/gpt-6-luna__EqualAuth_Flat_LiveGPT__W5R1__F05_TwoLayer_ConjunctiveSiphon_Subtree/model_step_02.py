import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _value(row, p):
    vmax, rho, K, c_loc = p
    u, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (K + u) + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[3.0, 0.5, 2.0, 0.05], bounds=([0.0, 0.0, 1e-6, -10.0], [100.0, 0.999, 100.0, 10.0]))
    return result.x

def predict(row):
    return float(_value(row, fit([dict(row, y=0.0)])))