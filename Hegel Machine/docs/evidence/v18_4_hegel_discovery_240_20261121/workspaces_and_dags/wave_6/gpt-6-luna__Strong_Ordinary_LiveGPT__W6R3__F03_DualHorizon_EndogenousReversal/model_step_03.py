import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        af, tf, ass, ts = p
        return u * (af * (1.0 - np.exp(-H / tf)) - ass * (1.0 - np.exp(-H / ts)))
    res = least_squares(lambda p: pred(p) - y, x0=[1.0, 1.0, 1.0, 20.0], bounds=([0.0, 1e-4, 0.0, 1e-4], [100.0, 100.0, 100.0, 1000.0]), max_nfev=50000)
    return res.x.tolist()

def predict(row):
    p = row['_fit_params']
    af, tf, ass, ts = p
    H = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-H / tf)) - ass * (1.0 - math.exp(-H / ts)))