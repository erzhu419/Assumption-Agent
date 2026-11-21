import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def features(row):
        u1 = float(row['u_1'])
        u2 = float(row['u_2'])
        return np.array([math.log1p(u1), u2 / (1.0 + u2)], dtype=float)

    X = np.vstack([features(row) for row in train_rows])
    y = np.array([float(row['y']) for row in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 1.0]))
    return {'coefficients': result.x.tolist()}

def predict(row):
    # Coefficients are set by fit(train_rows); attach them as module-level state.
    c = _model['coefficients']
    return float(c[0] * math.log1p(float(row['u_1'])) + c[1] * float(row['u_2']) / (1.0 + float(row['u_2'])))
