import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([1.0, math.log1p(u1), math.log1p(u2), u1*u2, u1*u1, u2*u2], dtype=float)


def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Linear coefficients in a compact smooth basis; selection is not a cause
    # in the full-population deployment target.
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    params = result.x
    def predict(row):
        return float(_features(row) @ params)
    return predict


def predict(row):
    # Fallback before fit; standard API use calls fit first.
    return 0.0
