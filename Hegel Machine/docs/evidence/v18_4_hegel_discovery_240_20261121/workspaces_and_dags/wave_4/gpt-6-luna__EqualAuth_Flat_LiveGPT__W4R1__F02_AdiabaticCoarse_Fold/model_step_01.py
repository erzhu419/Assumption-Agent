import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        k, a = p
        return np.array([r['y'] - (k*r['u_now'] + (r['x_now'] - k*r['u_now'])*math.exp(-a*r['H'])) for r in rows])
    result = least_squares(residual, x0=np.array([1.95, 1.0]), bounds=([0.0, 0.0], [10.0, 100.0]))
    return result.x

def predict(row):
    k, a = fit(_fit_rows)
    return float(k*row['u_now'] + (row['x_now'] - k*row['u_now'])*math.exp(-a*row['H']))