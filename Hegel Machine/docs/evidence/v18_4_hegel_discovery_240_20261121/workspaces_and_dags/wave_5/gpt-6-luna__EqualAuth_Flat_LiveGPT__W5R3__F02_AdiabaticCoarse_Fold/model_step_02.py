import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        k, a, b = p
        q = np.exp(-a*h)
        return k*u + (x-k*u)*q + b
    res = least_squares(lambda p: pred(p)-y, x0=[2.5, 2.0, 0.0], bounds=([0.0, 0.0, -10.0], [10.0, 100.0, 10.0]))
    return res.x

def predict(row):
    k, a, b = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    h = float(row['H'])
    return float(k*u + (x-k*u)*math.exp(-a*h) + b)
