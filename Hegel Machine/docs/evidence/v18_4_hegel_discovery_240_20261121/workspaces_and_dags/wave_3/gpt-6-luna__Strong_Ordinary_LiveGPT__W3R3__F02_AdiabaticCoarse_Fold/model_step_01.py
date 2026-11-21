import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    rows = train_rows
    def pred(p, r):
        k, a = p
        e = math.exp(-max(0.0, a) * r['H'])
        return k*r['u_now'] + (r['x_now'] - k*r['u_now'])*e
    def residual(p):
        return np.array([pred(p,r)-r['y'] for r in rows])
    res = least_squares(residual, x0=np.array([1.22, 1.0]), bounds=([-20.0, 0.0],[20.0, 100.0]))
    return res.x

def predict(row):
    p = fit(_train_rows)
    return float(p[0]*row['u_now'] + (row['x_now']-p[0]*row['u_now'])*math.exp(-p[1]*row['H']))