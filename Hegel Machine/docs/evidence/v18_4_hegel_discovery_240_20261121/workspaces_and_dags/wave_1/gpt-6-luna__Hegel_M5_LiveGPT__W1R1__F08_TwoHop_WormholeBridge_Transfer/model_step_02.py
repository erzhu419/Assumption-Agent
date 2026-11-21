import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_params(p, row):
        vmax, rho, c_loc = p
        u = max(float(row['u_now']), 0.0)
        s = float(row['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    def residuals(p):
        return np.array([predict_params(p, r) - float(r['y']) for r in rows])
    result = least_squares(residuals, x0=np.array([7.0, 0.7, 0.2]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return {'vmax': float(result.x[0]), 'rho': float(result.x[1]), 'c_loc': float(result.x[2])}

def predict(row):
    p = fit([row])
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return float(p['vmax'] * (1.0 - p['rho'] * s / (1.0 + s)) * (u / (1.2 + u)) + p['c_loc'] * math.sqrt(u))