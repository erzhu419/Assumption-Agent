import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(p, row):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(0.0, un))

def fit(train_rows):
    def residual(p):
        return np.array([_predict_params(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.2]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    return result.x

def predict(row):
    return float(_predict_params(_params, row))