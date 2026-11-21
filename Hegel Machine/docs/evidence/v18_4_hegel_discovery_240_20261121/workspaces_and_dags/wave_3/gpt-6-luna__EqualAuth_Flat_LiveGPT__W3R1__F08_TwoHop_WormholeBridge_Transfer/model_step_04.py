import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    un = u  # characteristic tau scale is 1.0
    def pred(p):
        vmax, rho, c_loc = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
        return base + c_loc * np.sqrt(un)
    res = least_squares(lambda p: pred(p) - y, x0=[8.0, 0.7, 0.2], bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
    return res.x

def predict(row):
    vmax, rho, c_loc = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * np.sqrt(un))