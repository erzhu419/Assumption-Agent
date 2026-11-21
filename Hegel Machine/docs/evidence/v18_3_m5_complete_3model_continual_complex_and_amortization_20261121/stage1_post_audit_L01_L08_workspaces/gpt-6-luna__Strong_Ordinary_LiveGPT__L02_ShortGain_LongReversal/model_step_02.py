import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    H = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        af, ass, tf, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    result = least_squares(lambda p: pred(p) - y, x0=[1.2, 1.0, 0.5, 8.0], bounds=([0.0, 0.0, 1e-4, 0.5001], [10.0, 10.0, 5.0, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    af, ass, tf, ts = fit([row]) if False else _params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts))))
