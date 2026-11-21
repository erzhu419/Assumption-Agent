import math
import numpy as np
from scipy.optimize import least_squares

_COEF = np.zeros(7)

def _features(row):
    indoor = float(row['u_indoor'])
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    impact = float(row['a_foot']) * float(row['impact_idx'])
    return np.array([1.0, indoor, indoor * indoor, outdoor, impact, impact * impact, indoor * impact], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    return float(_features(row) @ _COEF)
