import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['x_ind']) / (1.1 + float(r['x_ind'])), float(r['u_ctrl'])] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return X @ p - y
    result = least_squares(residual, x0=np.array([6.0, 0.3]))
    return result.x

def predict(row):
    vmax, c_ctrl = fit_parameters
    return float(vmax * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + c_ctrl * float(row['u_ctrl']))