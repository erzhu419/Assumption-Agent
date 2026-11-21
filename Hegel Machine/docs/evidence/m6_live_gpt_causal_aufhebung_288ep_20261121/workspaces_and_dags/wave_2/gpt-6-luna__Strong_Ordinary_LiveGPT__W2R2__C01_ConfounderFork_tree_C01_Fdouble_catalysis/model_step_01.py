import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) / 1.5 for r in train_rows])
    u = np.array([float(r.get('u_ctrl', 0.0)) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = x ** 2
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    p0 = np.array([8.0, 1.0, 0.1])
    result = least_squares(residual, p0, bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    vmax, k_half, c_ctrl = fit(_train_rows)
    x = float(row['x_ind']) / 1.5
    return float(vmax * (x ** 2) / (k_half + x ** 2) - c_ctrl * float(row.get('u_ctrl', 0.0)))
