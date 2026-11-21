import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor'])
    foot = float(row['a_foot'])
    impact = float(row['impact_idx'])
    return outdoor, indoor, foot, impact

def fit(train_rows):
    global _PARAMS
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    # Positive activity/space contributions with a nonlinear impact penalty.
    # Parameterization keeps the attenuation scale positive.
    def predict_params(p, row):
        outdoor, indoor, foot, impact = _features(row)
        a, b, c, d, log_k = p
        k = math.exp(log_k)
        activity = a * outdoor + b * indoor
        space = max(foot, 1e-6) ** c
        return activity * space * math.exp(-k * impact) + d

    def residual(p):
        return np.array([predict_params(p, r) for r in train_rows]) - y

    result = least_squares(residual, np.array([1.0, 1.0, 1.0, 0.0, 0.0]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return predict

def predict(row):
    outdoor, indoor, foot, impact = _features(row)
    a, b, c, d, log_k = _PARAMS
    return float((a * outdoor + b * indoor) * max(foot, 1e-6) ** c * math.exp(-math.exp(log_k) * impact) + d)
