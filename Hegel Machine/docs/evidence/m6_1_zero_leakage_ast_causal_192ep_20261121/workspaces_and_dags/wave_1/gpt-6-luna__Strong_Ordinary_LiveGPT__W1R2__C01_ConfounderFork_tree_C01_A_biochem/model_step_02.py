import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r.get('x_ind', 0.0)), float(r.get('u_ctrl', 0.0)), float(r.get('z_1', 0.0)), float(r.get('z_2', 0.0))] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Tree-indicator candidate: activation by the combined z1/z2 branch.
    ind = (X[:, 2] + X[:, 3] > 3.3).astype(float)
    def pred(p, x):
        vmax, kh, cctrl, bonus = p
        q = (x[:, 0] / 1.5) ** 2
        return vmax * q / (kh + q) - cctrl * x[:, 1] + bonus * ((x[:, 2] + x[:, 3]) > 3.3)
    def resid(p):
        return pred(p, X) - y
    res = least_squares(resid, x0=[8.0, 1.5, 0.1, 1.5], bounds=([0.01, 0.01, -10, -10], [100, 100, 10, 10]))
    return res.x.tolist()

def predict(row):
    p = fit([row])
    x = float(row.get('x_ind', 0.0)); u = float(row.get('u_ctrl', 0.0))
    q = (x / 1.5) ** 2
    return float(p[0] * q / (p[1] + q) - p[2] * u + p[3] * (float(row.get('z_1', 0.0)) + float(row.get('z_2', 0.0)) > 3.3))