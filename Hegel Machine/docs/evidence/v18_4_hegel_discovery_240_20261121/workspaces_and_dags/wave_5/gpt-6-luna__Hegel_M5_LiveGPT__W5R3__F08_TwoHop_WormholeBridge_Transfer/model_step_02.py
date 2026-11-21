import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def model(p):
        a, b, c, d = p
        return a + b * u / (c + u) + d * (1.0 - s)
    result = least_squares(lambda p: model(p) - y, x0=np.array([0.0, 6.0, 1.5, 1.0]), bounds=([-20.0, 0.0, 1e-6, -20.0], [20.0, 30.0, 30.0, 20.0]))
    return result.x.tolist()

def predict(row):
    a, b, c, d = fit(_TRAIN_ROWS)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a + b * u / (c + u) + d * (1.0 - s))