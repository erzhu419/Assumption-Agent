import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.0 + u2)

def fit(train_rows):
    X = np.asarray([_features(r) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([3.0, 4.0]))
    a, b = map(float, result.x)
    return {'a': a, 'b': b}

def predict(row):
    p = fit([row])
    return p['a'] * math.log1p(float(row['u_1'])) + p['b'] * float(row['u_2']) / (1.0 + float(row['u_2']))
