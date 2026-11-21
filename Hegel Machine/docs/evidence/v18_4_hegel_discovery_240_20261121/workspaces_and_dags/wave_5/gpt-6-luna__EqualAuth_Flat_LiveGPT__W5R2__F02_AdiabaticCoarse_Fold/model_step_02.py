import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        k, a = p
        return np.array([r['y'] - (k*r['u_now'] + (r['x_now'] - k*r['u_now'])*math.exp(-a*r['H'])) for r in rows])
    result = least_squares(residual, x0=np.array([2.1, 0.02]), bounds=([0.0, 0.0], [100.0, 100.0]))
    return tuple(float(v) for v in result.x)

def predict(row):
    k, a = fit([row])
    return k*row['u_now'] + (row['x_now'] - k*row['u_now'])*math.exp(-a*row['H'])