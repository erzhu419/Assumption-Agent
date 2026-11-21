import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x1 = np.array([float(r['u_1']) for r in train_rows])
    x2 = np.array([float(r['u_2']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, a, b):
        c, alpha, beta, k = p
        return c + alpha * np.log1p(a) + beta * b / (k + b)
    res = least_squares(lambda p: pred(p, x1, x2) - y,
                        x0=[0.0, 2.5, 2.0, 1.0],
                        bounds=([-20.0, -20.0, -20.0, 1e-6],
                                [20.0, 20.0, 20.0, 100.0]))
    return res.x.tolist()

def predict(row):
    p = fit([row]) if False else _fitted_params
    u1, u2 = float(row['u_1']), float(row['u_2'])
    c, alpha, beta, k = p
    return float(c + alpha * math.log1p(u1) + beta * u2 / (k + u2))