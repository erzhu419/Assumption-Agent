import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    x = np.array([float(r['x_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        k, a = p
        e = np.exp(-a * H)
        return k*u + (x-k*u)*e
    res = least_squares(lambda p: pred(p)-y, x0=np.array([2.5, 1.0]), bounds=([0.0, 0.0], [20.0, 20.0]))
    return {'k_ss': float(res.x[0]), 'a_eff': float(res.x[1])}

def predict(row):
    k = _MODEL['k_ss']
    a = _MODEL['a_eff']
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return float(k*u + (x-k*u)*math.exp(-a*H))

_MODEL = {'k_ss': 2.5, 'a_eff': 1.0}
_fit_impl = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _fit_impl(train_rows)
    return _MODEL