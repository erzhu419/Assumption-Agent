import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred_params(p, row):
        vmax, rho, c_loc = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred_params(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 0.6, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]))
    return result.x

def predict(row):
    p = fit(_training_rows)
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))