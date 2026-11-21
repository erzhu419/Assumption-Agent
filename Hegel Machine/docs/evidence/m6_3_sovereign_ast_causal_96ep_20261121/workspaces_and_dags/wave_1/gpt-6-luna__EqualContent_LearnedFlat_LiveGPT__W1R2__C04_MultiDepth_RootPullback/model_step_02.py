import numpy as np
from scipy.optimize import least_squares

_COEF = np.zeros(4)

def _features(row):
    return np.array([1.0, float(row['u_indoor']), float(row['u_outdoor']) * float(row['p_weather']), float(row['a_foot']) * float(row['impact_idx'])])

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    return float(_features(row) @ _COEF)
