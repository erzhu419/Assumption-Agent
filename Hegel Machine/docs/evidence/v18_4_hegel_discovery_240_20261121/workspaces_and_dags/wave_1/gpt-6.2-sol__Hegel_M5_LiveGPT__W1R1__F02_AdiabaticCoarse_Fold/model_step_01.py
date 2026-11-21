import math
import numpy as np
from scipy.optimize import least_squares

_gain = None

def fit(train_rows):
    global _gain
    u = np.asarray([row['u_now'] for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, x0=[1.0], bounds=(0.0, math.inf))
    _gain = float(result.x[0])

def predict(row) -> float:
    return float(_gain * row['u_now'])
