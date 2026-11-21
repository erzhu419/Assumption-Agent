import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] / 0.1 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, row):
        un = row['u_now'] / 0.1
        hill = un**2 / (p[1] + un**2)
        return p[0] * hill / (1.0 + p[2] * row['s_grad']) - p[3] * un
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-9, 0.0, 0.0], [100.0, 1e4, 1e3, 100.0]), max_nfev=20000)
    params = result.x
    def predictor(row):
        return float(pred(params, row))
    return predictor

def predict(row):
    return float('nan')
