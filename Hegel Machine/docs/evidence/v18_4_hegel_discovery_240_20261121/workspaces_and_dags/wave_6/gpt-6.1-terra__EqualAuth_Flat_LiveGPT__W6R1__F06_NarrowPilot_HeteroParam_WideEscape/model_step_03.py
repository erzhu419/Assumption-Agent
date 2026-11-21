import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return np.array([u / (1.2 + u), -s / (1.0 + s) * u / (1.2 + u), np.sqrt(u)])

def fit(train_rows):
    global _params
    X = np.array([_features(row) for row in train_rows])
    y = np.array([float(row['y']) for row in train_rows])
    _params = least_squares(lambda p: X @ p - y, np.ones(3)).x

def predict(row):
    return float(_features(row) @ _params)
