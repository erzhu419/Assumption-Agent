import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([7.5, 0.5]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    params = fit(_train_rows)
    return float(params['vmax'] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + params['c_ctrl'] * float(row['u_ctrl']))