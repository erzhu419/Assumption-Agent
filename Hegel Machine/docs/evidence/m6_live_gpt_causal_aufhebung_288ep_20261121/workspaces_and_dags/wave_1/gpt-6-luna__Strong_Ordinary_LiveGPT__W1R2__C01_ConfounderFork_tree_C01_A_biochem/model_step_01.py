import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, k_half, c_ctrl = p
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    q = (x / 1.5) ** 2
    return vmax * q / (k_half + q) - c_ctrl * u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    p0 = np.array([8.0, 1.0, 0.5])
    result = least_squares(lambda p: np.array([_pred(r, p) for r in train_rows]) - y,
                           p0, bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    # fit(train_rows) stores the fitted parameters on this function.
    return _pred(row, predict.params)
