import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        vmax, rho, c_loc = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    p0 = np.array([5.0, 0.5, 0.1])
    result = least_squares(residual, p0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    params = result.x.copy()
    def predict(row):
        return float(pred(params, row))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor')