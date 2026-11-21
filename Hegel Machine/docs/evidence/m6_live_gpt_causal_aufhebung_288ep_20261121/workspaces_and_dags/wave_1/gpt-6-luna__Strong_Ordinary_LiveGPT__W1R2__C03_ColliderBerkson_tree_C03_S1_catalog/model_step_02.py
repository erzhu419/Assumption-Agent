import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x1 = np.array([float(r['u_1']) for r in train_rows], dtype=float)
    x2 = np.array([float(r['u_2']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p, a, b):
        return p[0] * np.log1p(a) + p[1] * b / (p[2] + b)
    result = least_squares(lambda p: pred(p, x1, x2) - y,
                           x0=np.array([2.0, 4.0, 1.0]),
                           bounds=([0.0, 0.0, 1e-6], [20.0, 50.0, 50.0]))
    p = result.x.tolist()
    def predict(row):
        a = max(0.0, float(row['u_1']))
        b = max(0.0, float(row['u_2']))
        return float(p[0] * math.log1p(a) + p[1] * b / (p[2] + b))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')