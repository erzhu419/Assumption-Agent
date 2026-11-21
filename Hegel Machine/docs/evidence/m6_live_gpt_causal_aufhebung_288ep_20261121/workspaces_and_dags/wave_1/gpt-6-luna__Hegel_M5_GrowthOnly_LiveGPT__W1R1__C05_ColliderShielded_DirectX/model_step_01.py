import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    basis = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * basis + p[1] * u - y, x0=np.array([8.0, 0.5]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return fit_params['vmax'] * x / (1.1 + x) + fit_params['c_ctrl'] * u