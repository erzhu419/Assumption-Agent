import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, c_ctrl, b = p
        return vmax * x / (1.1 + x) + c_ctrl * u + b - y
    res = least_squares(residual, x0=[8.0, 0.8, 0.0])
    return {'vmax': float(res.x[0]), 'c_ctrl': float(res.x[1]), 'baseline': float(res.x[2])}

def predict(row):
    p = fit([row])
    return p['vmax'] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + p['c_ctrl'] * float(row['u_ctrl']) + p['baseline']