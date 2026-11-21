import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    return fit_params['vmax'] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + fit_params['c_ctrl'] * float(row['u_ctrl'])