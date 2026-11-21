import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    indoor = float(row.get('u_indoor', 0.0))
    outdoor = float(row.get('u_outdoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    impact = float(row.get('impact_idx', 0.0))
    area = float(row.get('a_foot', 0.0))
    return np.array([1.0, indoor, outdoor * weather, impact, area, indoor * impact], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda c: X @ c - y, np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(row) @ _COEF)