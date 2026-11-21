import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    s2 = u2 / (1.0 + u2)
    return np.array([math.log1p(u1), s2, math.log1p(u1) * s2], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda b: X @ b - y, x0=np.ones(X.shape[1]))
    return res.x

def predict(row):
    return float(_features(row) @ fit([{'u_1': r['u_1'], 'u_2': r['u_2'], 'y': r['y']} for r in _MODEL_ROWS]))