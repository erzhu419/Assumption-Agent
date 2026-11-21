import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    x = math.log1p(max(0.0, float(row['u_1'])))
    z = float(row['u_2'])
    return np.array([1.0, x, z, z*z, x*z], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    res = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]))
    return res.x.tolist()

def predict(row):
    # fit() coefficients are retained by the evaluation harness.
    b = _MODEL_COEF
    return float(_features(row) @ np.asarray(b, dtype=float))