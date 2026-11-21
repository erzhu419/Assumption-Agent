import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    X = np.array([[1.0, math.log1p(float(r['u_1'])), float(r['u_2'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda b: X @ b - y, x0=np.zeros(X.shape[1]))
    return result.x.tolist()


def predict(row):
    b = fit([{'u_1': 0.0, 'u_2': 0.0, 'y': 0.0}])
    return float(b[0] + b[1] * math.log1p(float(row['u_1'])) + b[2] * float(row['u_2']))