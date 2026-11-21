import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r.get('u_ctrl', 0.0)) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = (x / 1.5) ** 2
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.1], bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    vmax, k_half, c_ctrl = fit([row]) if False else _MODEL_PARAMS
    x = float(row['x_ind'])
    u = float(row.get('u_ctrl', 0.0))
    q = (x / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * u)
