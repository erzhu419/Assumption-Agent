import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.0]), bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
    return result.x

def predict(row):
    p = fit([row])
    return float(_value(row, p))