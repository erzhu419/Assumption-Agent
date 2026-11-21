import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([1.0, math.log1p(u1), u1, u2], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda beta: X @ beta - y, np.zeros(X.shape[1]))
    beta = result.x.copy()
    def predict(row):
        return float(_features(row) @ beta)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use the returned predictor.')