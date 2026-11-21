import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda a: X @ a - y, x0=np.array([1.0, 1.0]))
    return result.x

def predict(row):
    a = fit([row])
    return float(a[0] * math.log1p(float(row['u_1'])) + a[1] * float(row['u_2']) / (1.0 + float(row['u_2'])))