import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        k, a = p
        h = float(r['H'])
        return k*float(r['u_now']) + (float(r['x_now']) - k*float(r['u_now']))*math.exp(-a*h)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    opt = least_squares(residual, x0=[0.8, 1.0], bounds=([-10.0, 0.0], [10.0, 20.0]))
    return opt.x

def predict(row):
    k, a = fit([row])
    h = float(row['H'])
    u = float(row['u_now'])
    x = float(row['x_now'])
    return float(k*u + (x-k*u)*math.exp(-a*h))