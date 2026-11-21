import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    p = float(row['u_past'])
    n = float(row['u_now'])
    h = float(row['H'])
    # Normalized before/after contrast; the small denominator floor keeps
    # predictions defined if either input is zero.
    contrast = (p - n) / max(abs(p) + abs(n), 1e-12)
    return np.array([1.0, contrast, h, contrast * h], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]))
    beta = result.x
    def predict(row):
        return float(_features(row) @ beta)
    return predict

def predict(row):
    # Default fitted parameters are not meaningful without training data;
    # callers should use the predictor returned by fit(train_rows).
    return 0.0
