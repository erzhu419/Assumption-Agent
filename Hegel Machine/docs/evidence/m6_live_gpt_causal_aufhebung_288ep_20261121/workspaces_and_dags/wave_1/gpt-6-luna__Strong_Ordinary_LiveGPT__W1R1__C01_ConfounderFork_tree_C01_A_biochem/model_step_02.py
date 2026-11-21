import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    z = np.array([float(r['z_2']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = (z / 1.5) ** 2
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    result = least_squares(residual, x0=[9.0, 1.0, 0.5], bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    fit.params = result.x
    return fit

def predict(row):
    vmax, k_half, c_ctrl = fit.params
    q = (float(row['z_2']) / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))