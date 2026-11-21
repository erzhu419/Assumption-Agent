import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    xs = np.array([float(r['x_ind']) / (1.1 + float(r['x_ind'])) for r in train_rows])
    us = np.array([float(r['u_ctrl']) for r in train_rows])
    ys = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: p[0] * xs + p[1] * us - ys, x0=np.array([1.0, 1.0]))
    return result.x

def predict(row):
    p = fit(_train_rows)
    x = float(row['x_ind'])
    return float(p[0] * x / (1.1 + x) + p[1] * float(row['u_ctrl']))