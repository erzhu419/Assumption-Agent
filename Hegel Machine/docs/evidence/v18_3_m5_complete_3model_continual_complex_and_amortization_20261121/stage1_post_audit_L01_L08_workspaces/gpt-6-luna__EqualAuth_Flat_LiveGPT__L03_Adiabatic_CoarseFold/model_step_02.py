import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    x = np.array([float(r['x_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        c, k, loga = p
        a = math.exp(loga)
        e = np.exp(-a * H)
        return c + k*u + (x - k*u)*e
    res = least_squares(lambda p: pred(p)-y, x0=np.array([0.0, 2.0, 0.0]), bounds=([-20.0,-20.0,-8.0],[20.0,20.0,5.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    c, k, loga = fit_globals
    a = math.exp(loga)
    e = math.exp(-a * float(row['H']))
    u = float(row['u_now'])
    x = float(row['x_now'])
    return c + k*u + (x-k*u)*e

fit_globals = (0.0, 2.0, 0.0)