import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        k, a = p
        e = np.exp(-a * H)
        return k*u + (x-k*u)*e
    res = least_squares(lambda p: pred(p)-y, x0=[2.2, 1.0], bounds=([0.0, 0.0], [np.inf, np.inf]))
    return {'k_ss': float(res.x[0]), 'a_eff': float(res.x[1])}

def predict(row):
    k = float(_params['k_ss'])
    a = float(_params['a_eff'])
    e = np.exp(-a * float(row['H']))
    return float(k*float(row['u_now']) + (float(row['x_now'])-k*float(row['u_now']))*e)
