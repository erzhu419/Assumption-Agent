import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    result = least_squares(lambda p: pred(p) - y, x0=np.array([6.0, 0.5, 0.0]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return {'vmax': float(result.x[0]), 'rho': float(result.x[1]), 'c_loc': float(result.x[2])}

def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(p['vmax'] * (1.0 - p['rho'] * s / (1.0 + s)) * (u / (1.2 + u)) + p['c_loc'] * math.sqrt(u))