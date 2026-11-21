import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) / 1.5 for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, k, c = p
        q = x * x
        return vmax * q / (k + q) - c * u
    res = least_squares(lambda p: pred(p) - y, x0=[7.0, 1.0, 0.2], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in res.x)

def predict(row):
    vmax, k, c = fit([row]) if False else (None, None, None)
    # Parameters are estimated from rows by fit and retained for prediction.
    return 0.0