import numpy as np
from scipy.optimize import least_squares

_scale = None

def fit(train_rows):
    global _scale
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, x0=[1.0])
    _scale = float(result.x[0])

def predict(row):
    return float(_scale * row['u_now'])
