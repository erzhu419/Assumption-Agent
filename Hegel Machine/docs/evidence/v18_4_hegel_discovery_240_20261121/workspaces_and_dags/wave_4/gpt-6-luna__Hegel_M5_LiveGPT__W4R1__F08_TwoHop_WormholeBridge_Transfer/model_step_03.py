import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        a, b, c = p
        return a * u / (b + u) * np.exp(-c * (s - 1.0))
    res = least_squares(lambda p: pred(p) - y, x0=[4.0, 1.2, 0.3], bounds=([0.001, 0.001, -3.0], [20.0, 20.0, 3.0]))
    return res.x.tolist()

def predict(row):
    a, b, c = fit(_TRAIN_ROWS)
    return float(a * row['u_now'] / (b + row['u_now']) * np.exp(-c * (row['s_grad'] - 1.0)))