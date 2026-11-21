import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    indoor = float(row['u_indoor'])
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    foot = float(row['a_foot'])
    impact = float(row['impact_idx'])
    burden = foot * impact
    # Compact nonlinear response basis for indoor activity, weather-scaled
    # outdoor activity, and footprint/impact burden, including interactions.
    return np.array([
        1.0, indoor, indoor**2, indoor**3,
        outdoor, burden, burden**2, burden**3,
        indoor*burden, indoor**2*burden,
        indoor*burden**2, outdoor*indoor, outdoor*burden
    ], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Mild ridge penalty stabilizes the nonlinear basis without penalizing intercept.
    penalty = np.diag([0.0] + [1e-5] * (X.shape[1] - 1))
    result = least_squares(lambda b: np.concatenate([X @ b - y, penalty @ b]), np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    return float(_features(row) @ _COEF)
