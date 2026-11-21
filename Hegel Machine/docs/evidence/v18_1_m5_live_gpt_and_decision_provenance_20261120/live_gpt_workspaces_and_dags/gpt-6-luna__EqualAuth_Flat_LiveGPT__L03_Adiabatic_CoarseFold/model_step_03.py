import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # y = k*u + (x-k*u)*exp(-a*H)
    def pred(p, r):
        k, a = p
        e = math.exp(-a * r['H'])
        return k*r['u_now'] + (r['x_now']-k*r['u_now'])*e
    def residual(p):
        return [pred(p,r)-r['y'] for r in train_rows]
    res = least_squares(residual, [1.0, 0.2], bounds=([0.0, 0.0],[10.0, 10.0]))
    return res.x

def predict(row):
    # Parameters are fitted by fit and attached to function for API compatibility.
    k, a = predict.params
    e = math.exp(-a * row['H'])
    return float(k*row['u_now'] + (row['x_now']-k*row['u_now'])*e)
