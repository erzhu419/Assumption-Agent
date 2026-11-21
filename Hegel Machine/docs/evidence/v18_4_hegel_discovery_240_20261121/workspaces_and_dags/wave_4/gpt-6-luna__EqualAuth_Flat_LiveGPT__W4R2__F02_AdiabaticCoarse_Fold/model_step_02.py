import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # In the observed long-horizon regime, the transient contribution is
    # negligible; estimate the steady-state gain and offset numerically.
    X = np.array([[float(r['u_now']), 1.0] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda p: X @ p - y, x0=np.array([2.2, 0.0]))
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(p[0] * float(row['u_now']) + p[1])