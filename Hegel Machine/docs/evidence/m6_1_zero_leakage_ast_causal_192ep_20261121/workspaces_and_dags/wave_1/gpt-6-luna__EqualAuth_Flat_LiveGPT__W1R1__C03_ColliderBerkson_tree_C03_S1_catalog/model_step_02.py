import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([math.log1p(u1), u2 / (1.5 + u2)], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 1.0]))
    params = result.x.copy()
    def predict(row):
        return float(_features(row) @ params)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')