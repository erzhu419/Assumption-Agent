import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    H = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        af, tf, ass, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    res = least_squares(lambda p: pred(p) - y, x0=[1.0, 0.5, 0.5, 10.0], bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 100.0, 20.0, 500.0]), max_nfev=20000)
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    H = float(row['H'])
    af, tf, ass, ts = p
    return float(u * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts))))