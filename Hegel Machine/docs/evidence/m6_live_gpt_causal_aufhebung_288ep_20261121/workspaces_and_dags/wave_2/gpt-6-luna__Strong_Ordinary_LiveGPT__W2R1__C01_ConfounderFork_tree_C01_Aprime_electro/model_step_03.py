import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _params
    x = np.array([float(r['z_1']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, xx, uu):
        vmax, k_half, c_ctrl = p
        q = (xx / 1.5) ** 2
        return vmax * q / (k_half + q) - c_ctrl * uu
    p0 = [max(y) + 1.0, 1.0, 0.2]
    res = least_squares(lambda p: pred(p, x, u) - y, p0, bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _params = res.x
    return _params

def predict(row):
    vmax, k_half, c_ctrl = _params
    x = float(row['z_1'])
    q = (x / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))