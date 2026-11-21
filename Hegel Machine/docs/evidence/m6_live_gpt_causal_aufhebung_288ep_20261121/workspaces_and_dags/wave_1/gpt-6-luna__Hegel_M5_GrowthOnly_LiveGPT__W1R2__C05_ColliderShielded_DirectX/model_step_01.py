import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([6.0, 0.35]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    vmax = float(fit_params['vmax'])
    c_ctrl = float(fit_params['c_ctrl'])
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return vmax * x / (1.1 + x) + c_ctrl * u