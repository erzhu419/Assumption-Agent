import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        c, a, b, d, e = p
        return c + a * u / (b + u) + d * (s - 1.0) + e * (s - 1.0) ** 2
    result = least_squares(lambda p: pred(p) - y, x0=np.array([0.0, 6.0, 2.0, -0.5, 0.0]), bounds=([-100.0, -100.0, 0.001, -100.0, -100.0], [100.0, 100.0, 100.0, 100.0, 100.0]))
    return result.x

def predict(row):
    c, a, b, d, e = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(c + a * u / (b + u) + d * (s - 1.0) + e * (s - 1.0) ** 2)
