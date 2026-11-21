import math
import numpy as np
from scipy.optimize import least_squares

_gain = None

def fit(train_rows):
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    global _gain
    _gain = float(least_squares(lambda p: p[0] * u - y, [1.0]).x[0])

def predict(row) -> float:
    return float(_gain * row['u_now'])
